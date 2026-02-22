import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class LayerNorm(nn.LayerNorm):
    """fp16-safe LayerNorm (CLIP-style)"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)


class TextEncoder(nn.Module):
    """
    CLIP text encoder wrapper.
    - return pooled (EOT) feature by default
    - optionally return full projected token sequence + eot_idx
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_tokens: bool = False):
        # prompts: (B, L, C)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)               # (L, B, C)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)               # (B, L, C)
        x = self.ln_final(x).type(self.dtype)

        tokens_proj = x @ self.text_projection                       # (B, L, D=512)
        eot_idx = tokenized_prompts.argmax(dim=-1)                   # (B,)
        pooled = tokens_proj[torch.arange(tokens_proj.size(0), device=tokens_proj.device), eot_idx]  # (B, 512)

        if return_tokens:
            return pooled, tokens_proj, eot_idx
        return pooled


class InversionNetwork(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, v):
        x = self.act(self.fc1(v))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x


class PromptComposer(nn.Module):
    def __init__(self, clip_model, prompt_mode: str):
        super().__init__()
        self.prompt_mode = prompt_mode
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        self.composed_str = "A photo of a X person"
        self.simplified_str = "A photo of a person"

        # register buffers once (empty => chưa khởi tạo)
        self.register_buffer("tokenized_composed", torch.empty(0, dtype=torch.long))
        self.register_buffer("tokenized_simplified", torch.empty(0, dtype=torch.long))
        self.register_buffer("embed_composed", torch.empty(0))
        self.register_buffer("embed_simplified", torch.empty(0))

        self.x_pos = None

    def _ensure_tokenization(self):
        if self.tokenized_composed.numel() == 0:
            import model.clip.clip as clip_module

            dev = self.token_embedding.weight.device  # cùng device với embedding

            tokenized_composed = clip_module.tokenize([self.composed_str]).to(dev)
            tokenized_simplified = clip_module.tokenize([self.simplified_str]).to(dev)

            tokenized_x = clip_module.tokenize(["X"]).to(dev)
            x_token_id = tokenized_x[0, 1].item()

            x_pos = (tokenized_composed[0] == x_token_id).nonzero(as_tuple=False)
            if x_pos.numel() == 0:
                raise ValueError("Cannot locate placeholder token in composed prompt")

            # chỉ gán (buffer đã tồn tại từ __init__)
            self.tokenized_composed = tokenized_composed
            self.tokenized_simplified = tokenized_simplified
            self.x_pos = int(x_pos[0].item())

    def _ensure_embeddings(self):
        self._ensure_tokenization()
        if self.embed_composed.numel() == 0:
            with torch.no_grad():
                embed_composed = self.token_embedding(self.tokenized_composed).type(self.dtype)
                embed_simplified = self.token_embedding(self.tokenized_simplified).type(self.dtype)

            self.embed_composed = embed_composed
            self.embed_simplified = embed_simplified

    def forward(self, s_star: torch.Tensor):
        self._ensure_embeddings()
        b = s_star.shape[0]
        if self.prompt_mode == 'simplified':
            tokenized = self.tokenized_simplified.expand(b, -1)
            prompts = self.embed_simplified.expand(b, -1, -1)
            return prompts, tokenized

        s_star = s_star.to(dtype=self.embed_composed.dtype)

        tokenized = self.tokenized_composed.expand(b, -1)
        prefix = self.embed_composed[:, :self.x_pos, :].expand(b, -1, -1)
        suffix = self.embed_composed[:, self.x_pos + 1:, :].expand(b, -1, -1)
        prompts = torch.cat([prefix, s_star.unsqueeze(1), suffix], dim=1)
        return prompts, tokenized


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# =========================
# 3 CLASSES REPLACED HERE:
#   - CrossAttention  (built-in nn.MultiheadAttention)
#   - SelfAttention   (built-in nn.MultiheadAttention)
#   - TransformerBlock (SelfAttn + FFN)
# Everything else kept the same.
# =========================

class CrossAttention(nn.Module):
    """
    Built-in cross-attention using nn.MultiheadAttention (batch_first=True).
    Keep the SAME init signature as your old CrossAttention so the rest of code stays unchanged.

    forward signature kept compatible with existing calls:
      out, attn_w = self.cross_attn(q, k, v, need_weights=True)

    Shapes:
      q: (B, Nq, D)
      k: (B, Nk, D)
      v: (B, Nk, D)
      attn_w: (B, H, Nq, Nk)  (per-head weights)
    """
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = embedding_dim // downsample_rate

        assert self.internal_dim % num_heads == 0, "num_heads must divide internal_dim."

        # Optional downsample projections (keep compatibility with downsample_rate)
        if downsample_rate != 1:
            self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
            self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
            self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
            mha_dim = self.internal_dim
            self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            mha_dim = embedding_dim
            self.out_proj = None

        self.mha = nn.MultiheadAttention(
            embed_dim=mha_dim,
            num_heads=num_heads,
            dropout=dropout,          # dropout on attn weights
            batch_first=True,         # (B,N,D)
        )

    def forward(self, q, k, v, need_weights: bool = False):
        if self.q_proj is not None:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        out, attn_w = self.mha(
            query=q,
            key=k,
            value=v,
            need_weights=need_weights,
            average_attn_weights=False,   # keep per-head weights: (B,H,Nq,Nk)
        )

        if self.out_proj is not None:
            out = self.out_proj(out)

        if need_weights:
            return out, attn_w
        return out


class SelfAttention(nn.Module):
    """
    Built-in self-attention using nn.MultiheadAttention (batch_first=True).
    Keep SAME init signature as old SelfAttention for minimal changes.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        # qkv_bias kept for compatibility (MultiheadAttention has bias=True by default)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True
        )

    def forward(self, x):
        out, _ = self.mha(
            query=x, key=x, value=x,
            need_weights=False
        )
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (Pre-LN):
      x = x + DropPath(SelfAttn(LN(x)))
      x = x + DropPath(FFN(LN(x)))

    (FFN uses built-in Linear/Dropout + chosen activation)
    """
    def __init__(self, d_model=512, nhead=8, mlp_ratio=4.0, drop_path=0.0,
                 attn_drop=0.0, proj_drop=0.0, act_layer=QuickGELU):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = SelfAttention(dim=d_model, num_heads=nhead, qkv_bias=True, dropout=attn_drop)
        self.norm2 = LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(proj_drop),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# =========================
# Everything below kept the same (except PostCABlock now just uses TransformerBlock).
# =========================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=QuickGELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PostCABlock(TransformerBlock):
    """
    Backward-compatible name for your code.
    Now implemented by TransformerBlock (SelfAttn + FFN).
    """
    pass


class MultimodalInteractionModule(nn.Module):
    """
    MIM with:
    - explicit cross-attn (no MultiheadAttention)
    - text query mode: full tokens OR EOT token
    - attn_map mode:
        * "mean_head_mean_text_norm": mean heads + mean text_len + normalize sum=1
        * "mean_head": mean heads only; if Nq>1 -> pick EOT row (no mean text_len)
    - reweight: mul_mean1 / mul / residual
    - gradients flow back to inversion (NO detach in model forward)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        reweight: str = "mul_mean1",               # "mul_mean1" | "mul" | "residual"
        text_query_mode: str = "eot",              # "full" | "eot"
        kv_mode: str = "patch",                  # "patch" | "cls_patch"
        post_seq_mode: str = "image",             # "image" | "text_image" | "cls_patch" | "cls_text" | "cls_patch_text"
        attn_map_mode: str = "mean_head_mean_text_norm",  # "mean_head_mean_text_norm" | "mean_head"
        attn_pool_mode: str = "mean",              # "mean" | "max" - pooling over heads
        eps: float = 1e-6,
        act_layer=QuickGELU,
    ):
        super().__init__()
        self.reweight = reweight
        self.text_query_mode = text_query_mode
        self.kv_mode = kv_mode
        self.post_seq_mode = post_seq_mode
        self.attn_map_mode = attn_map_mode
        self.attn_pool_mode = attn_pool_mode
        self.eps = eps

        self.cross_attn = CrossAttention(embedding_dim=embed_dim, num_heads=num_heads, dropout=attn_drop)

        self.q_ln = LayerNorm(embed_dim)
        self.kv_ln = LayerNorm(embed_dim)

        self.post_blocks = nn.ModuleList([
            PostCABlock(
                d_model=embed_dim,
                nhead=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
            )
            for _ in range(int(num_blocks))
        ])

    def _build_query(self, text_tokens_full: torch.Tensor, eot_idx: torch.Tensor):
        # text_tokens_full: (B, L, D)
        if self.text_query_mode == "full":
            return text_tokens_full  # (B, L, D)
        elif self.text_query_mode == "eot":
            B = text_tokens_full.size(0)
            q = text_tokens_full[torch.arange(B, device=text_tokens_full.device), eot_idx]  # (B, D)
            return q.unsqueeze(1)  # (B,1,D)
        else:
            raise ValueError(f"Unknown text_query_mode: {self.text_query_mode}")

    def _pool_query(self, seq: torch.Tensor, eot_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Pool query sequence -> a single vector (B,D)
        - if Nq == 1: return that token
        - else if eot_idx is provided: pick EOT token
        - else: mean pool
        """
        # seq: (B, Nq, D)
        if seq.dim() != 3:
            raise ValueError(f"seq must be (B,Nq,D), got {seq.shape}")

        B, Nq, D = seq.shape
        if Nq == 1:
            return seq[:, 0, :]  # (B,D)

        if eot_idx is not None:
            # eot_idx: (B,)
            idx = eot_idx.to(seq.device).long().clamp(0, Nq - 1)  # safety clamp
            # gather along token dimension
            return seq.gather(1, idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)  # (B,D)

        # fallback
        return seq.mean(dim=1)

    def _make_attn_map(self, attn_w: torch.Tensor, eot_idx: torch.Tensor):
        # attn_w: (B, H, Nq, M)
        B, H, Nq, M = attn_w.shape

        pool_fn = lambda x, dim: x.mean(dim=dim) if self.attn_pool_mode == "mean" else x.max(dim=dim)[0]

        if self.attn_map_mode == "mean_head_mean_text_norm":
            attn_map = pool_fn(attn_w, dim=1)                  # (B, Nq, M)
            attn_map = attn_map.mean(dim=1, keepdim=True)      # (B, 1, M)
            attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + self.eps)
            return attn_map

        if self.attn_map_mode == "mean_head":
            # pool heads only
            attn_map = pool_fn(attn_w, dim=1)  # (B, Nq, M)
            if attn_map.size(1) > 1:
                # keep "no mean text_len": pick EOT row
                attn_map = attn_map[torch.arange(B, device=attn_map.device), eot_idx].unsqueeze(1)  # (B,1,M)
            # else already (B,1,M)
            return attn_map

        raise ValueError(f"Unknown attn_map_mode: {self.attn_map_mode}")

    def _reweight_patches(self, patch_tokens: torch.Tensor, attn_map: torch.Tensor):
        # patch_tokens: (B, M, D), attn_map: (B, 1, M)
        B, M, D = patch_tokens.shape
        w = attn_map.transpose(1, 2)  # (B, M, 1)

        if self.reweight == "mul_mean1":
            scale = w * M
            return patch_tokens * scale
        elif self.reweight == "mul":
            return patch_tokens * w
        elif self.reweight == "residual":
            scale = w * M
            return patch_tokens * (1.0 + scale)
        else:
            raise ValueError(f"Unknown reweight mode: {self.reweight}")

    def forward(self, text_tokens_full, eot_idx, patch_tokens, cls_token, return_cls_states: bool = False):
        """
        PromptSG-style path (paper-faithful):
          Z = CrossAttn(Q(text_full_or_eot), K/V(image_tokens))   # Eq.(7)
          Z -> (SelfAttn + FFN) x num_blocks                      # post_blocks
        Notes:
          - We DO NOT reweight/aggregate patch tokens into a new patch sequence here.
          - We still return a pooled attention map over patches for visualization.
        Inputs:
          text_tokens_full: (B, L, D)
          eot_idx:          (B,)
          patch_tokens:     (B, M, D)
          cls_token:        (B, 1, D)
        Returns:
          seq: (B, Nq, D)   refined query sequence after post blocks
          patch_attn_map: (B, 1, M) pooled attention over patches
          cls_states (optional): list[(B,D)] pooled vector after each block (from seq)
        """
        # LN
        text_tokens_full = self.q_ln(text_tokens_full)

        # Query selection (full / eot)
        q = self._build_query(text_tokens_full, eot_idx=eot_idx)  # (B, Nq, D)

        # KV selection (patch / cls_patch)
        if self.kv_mode == "patch":
            kv_tokens = patch_tokens                                   # (B, M, D)
        elif self.kv_mode == "cls_patch":
            kv_tokens = torch.cat([cls_token, patch_tokens], dim=1)    # (B, 1+M, D)
        elif self.kv_mode == "cls":
            kv_tokens = cls_token                                      # (B, 1, D)
        else:
            raise ValueError(f"Unknown kv_mode: {self.kv_mode}")

        kv = self.kv_ln(kv_tokens)

        # Eq.(7): cross-attn output is the query-side sequence Z
        seq, attn_w = self.cross_attn(q, kv, kv, need_weights=True)     # seq: (B,Nq,D), attn_w: (B,H,Nq,Nkv)

        # Attention map for visualization (pool heads + optionally pool query tokens)
        attn_map = self._make_attn_map(attn_w, eot_idx=eot_idx)         # (B,1,Nkv)

        # Slice patch-only map (drop CLS column if present)
        if self.kv_mode == "cls_patch":
            patch_attn_map = attn_map[:, :, 1:]                         # (B,1,M)
            if self.attn_map_mode == "mean_head_mean_text_norm":
                patch_attn_map = patch_attn_map / (patch_attn_map.sum(dim=-1, keepdim=True) + self.eps)
        elif self.kv_mode == "cls":
            # Khi chỉ dùng CLS token, không có patch tokens -> patch_attn_map có thể để None hoặc giữ nguyên (B,1,1)
            patch_attn_map = attn_map  # (B,1,1) – không dùng cho patch visualization
        else:
            patch_attn_map = attn_map                                   # (B,1,M)

        # Post blocks operate on the query sequence (SelfAttn + FFN)
        cls_states = [self._pool_query(seq, eot_idx=eot_idx)]
        for blk in self.post_blocks:
            seq = blk(seq)
            cls_states.append(self._pool_query(seq, eot_idx=eot_idx))

        if return_cls_states:
            return seq, patch_attn_map, cls_states
        return seq, patch_attn_map


# -----------------------------------------------------------------------------
# CLIP-SCGI modules (arXiv:2410.09382)
# -----------------------------------------------------------------------------
class CaptionGuidedInversion(nn.Module):
    """Caption-Guided Inversion (CGI).

    - Encode caption -> cap_feat (B,512)
    - Refine visual tokens with element-wise multiplication
    - Build pseudo-word s* from global (CLS) + local (query->patch cross-attn)
    """
    def __init__(self, embed_dim: int = 512, num_queries: int = 2, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.q_ln = LayerNorm(embed_dim)
        self.kv_ln = LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # simple mappings to token space
        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor, cap_feat: torch.Tensor):
        """Args:
            cls_token:   (B,512)
            patch_tokens:(B,M,512)
            cap_feat:    (B,512)
        Returns:
            s_star:      (B,512)
            cls_ref:     (B,512)
            patch_ref:   (B,M,512)
        """
        # refine visual with caption embedding
        cls_ref = cls_token * cap_feat
        patch_ref = patch_tokens * cap_feat.unsqueeze(1)

        # local branch: learnable queries attend to refined patches
        B = cls_token.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B,K,512)
        q = self.q_ln(q)
        kv = self.kv_ln(patch_ref)

        local, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B,K,512)
        local_pool = local.mean(dim=1)  # (B,512)

        g = self.global_mlp(cls_ref)
        l = self.local_mlp(local_pool)
        s_star = self.fuse(torch.cat([g, l], dim=-1))
        return s_star, cls_ref, patch_ref


class CLIPSCGIModel(nn.Module):
    """CLIP-SCGI for Person ReID.

    Training:
      - Requires offline captions (Ik, Tk)
      - CGI: caption-guided inversion to build s* and refine image tokens
      - CFF: cross-attn (Q=text tokens) with K/V=image tokens + 2 transformer blocks
      - Loss: L_id + L_tri + L_con (paired InfoNCE)

    Inference:
      - Drop caption-guided branch
      - Use fixed prompt: "A photo of a person"
      - Final embedding from last layer of CFF
    """
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = num_classes

        # CLIP visual output dim (ViT projected dim = 512)
        self.in_planes_proj = 512

        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # Load CLIP model (same as PromptSG)
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        # composers
        self.prompt_composer_composed = PromptComposer(clip_model, "composed")
        self.prompt_composer_simplified = PromptComposer(clip_model, "simplified")

        # CGI
        cgi_cfg = getattr(cfg.MODEL, "CLIPSCGI", None)
        num_q = getattr(cgi_cfg, "CGI_NUM_QUERIES", 2) if cgi_cfg is not None else 2
        cgi_heads = getattr(cgi_cfg, "CGI_HEADS", 8) if cgi_cfg is not None else 8
        self.cgi = CaptionGuidedInversion(embed_dim=512, num_queries=num_q, num_heads=cgi_heads)

        # CFF (reuse MIM implementation: query-side cross-attn + post transformer blocks)
        cff_heads = getattr(cgi_cfg, "CFF_HEADS", 8) if cgi_cfg is not None else 8
        cff_blocks = getattr(cgi_cfg, "CFF_POST_BLOCKS", 2) if cgi_cfg is not None else 2
        text_mode = getattr(cgi_cfg, "COATTN_TEXT_MODE", "full") if cgi_cfg is not None else "full"
        kv_mode = getattr(cgi_cfg, "KV_MODE", "cls_patch") if cgi_cfg is not None else "cls_patch"

        self.cff = MultimodalInteractionModule(
            embed_dim=512,
            num_heads=cff_heads,
            num_blocks=cff_blocks,
            reweight="mul_mean1",
            text_query_mode=text_mode,
            attn_map_mode="mean_head_mean_text_norm",
            attn_pool_mode="mean",
            kv_mode=kv_mode,
            post_seq_mode="cls_patch",
            act_layer=QuickGELU,
        )

        # Freeze text encoder weights (but gradients must pass through to s*)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.token_embedding.parameters():
            p.requires_grad_(False)
        self.text_encoder.eval()

        self.drop_cgi_in_infer = getattr(cgi_cfg, "DROP_CGI_IN_INFER", True) if cgi_cfg is not None else True
        self.use_caption = getattr(cgi_cfg, "USE_CAPTION", True) if cgi_cfg is not None else True

    @torch.no_grad()
    def _encode_caption(self, captions, device):
        """Encode raw caption strings into CLIP text features (pooled EOT)."""
        import model.clip.clip as clip_module
        tokenized = clip_module.tokenize(list(captions)).to(device)
        prompts = self.token_embedding(tokenized).type(self.dtype)
        cap_feat = self.text_encoder(prompts, tokenized)  # (B,512)
        return cap_feat

    def forward(self, x=None, label=None, caption=None, skip_mim: bool = False, **kwargs):
        # Visual tokens (projected to 512)
        if self.model_name != "ViT-B-16":
            raise NotImplementedError("CLIP-SCGI path currently implemented for ViT-B-16 only.")
        _, _, xproj = self.image_encoder(x)  # xproj: (B,1+M,512)

        cls_token = xproj[:, 0]      # (B,512)
        patch_tokens = xproj[:, 1:]  # (B,M,512)

        device = xproj.device
        B = xproj.size(0)

        # Inference: drop CGI and use fixed prompt
        if (not self.training) and self.drop_cgi_in_infer:
            # fixed prompt: "A photo of a person"
            prompts, tokenized_prompt = self.prompt_composer_simplified(torch.zeros(B, 512, device=device, dtype=xproj.dtype))
            pooled_text, text_tokens_full, eot_idx = self.text_encoder(prompts, tokenized_prompt, return_tokens=True)

            if skip_mim:
                feat = cls_token
            else:
                seq, _ = self.cff(
                    text_tokens_full=text_tokens_full,
                    eot_idx=eot_idx,
                    patch_tokens=patch_tokens,
                    cls_token=cls_token.unsqueeze(1),
                    return_cls_states=False,
                )
                if seq.size(1) == 1:
                    feat = seq[:, 0, :]
                else:
                    feat = seq[torch.arange(B, device=device), eot_idx, :]

            bn_feat = self.bottleneck_proj(feat)

            if not self.training:
                return bn_feat if self.neck_feat == "after" else feat

        # Training: need captions to build s*
        cap_feat = None
        if self.use_caption and (caption is not None):
            if isinstance(caption, (list, tuple)):
                caption = [c if c is not None else "" for c in caption]
            cap_feat = self._encode_caption(caption, device=device)

        # If captions missing, fallback to simplified prompt (still trainable but not CLIP-SCGI)
        if cap_feat is None:
            prompts, tokenized_prompt = self.prompt_composer_simplified(torch.zeros(B, 512, device=device, dtype=xproj.dtype))
            pooled_text, text_tokens_full, eot_idx = self.text_encoder(prompts, tokenized_prompt, return_tokens=True)
            cls_ref, patch_ref = cls_token, patch_tokens
            s_star = None
        else:
            s_star, cls_ref, patch_ref = self.cgi(cls_token, patch_tokens, cap_feat)
            prompts, tokenized_prompt = self.prompt_composer_composed(s_star)
            pooled_text, text_tokens_full, eot_idx = self.text_encoder(prompts, tokenized_prompt, return_tokens=True)

        # CFF
        if skip_mim:
            feat = cls_ref
        else:
            seq, _ = self.cff(
                text_tokens_full=text_tokens_full,
                eot_idx=eot_idx,
                patch_tokens=patch_ref,
                cls_token=cls_ref.unsqueeze(1),
                return_cls_states=False,
            )
            if seq.size(1) == 1:
                feat = seq[:, 0, :]
            else:
                feat = seq[torch.arange(B, device=device), eot_idx, :]

        bn_feat = self.bottleneck_proj(feat)
        cls_score = self.classifier_proj(bn_feat)

        # For CLIP-SCGI contrastive: use refined CLS and caption feature (paired)
        image_contrast_feat = cls_ref
        text_contrast_feat = cap_feat

        # Return signature compatible with processor/loss:
        #   cls_score, triplet_feats, image_feat, text_feat
        return cls_score, feat, image_contrast_feat, text_contrast_feat



class PromptSGModel(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.prompt_mode = cfg.MODEL.PROMPTSG.PROMPT_MODE
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num

        # config knobs
        self.coattn_text_mode = getattr(cfg.MODEL.PROMPTSG, "COATTN_TEXT_MODE", "full")  # "full"|"eot"
        self.attn_map_mode = getattr(cfg.MODEL.PROMPTSG, "ATTN_MAP_MODE", "mean_head_mean_text_norm")
        self.reweight_mode = getattr(cfg.MODEL.PROMPTSG, "REWEIGHT_MODE", "mul_mean1")
        self.kv_mode = getattr(cfg.MODEL.PROMPTSG, "KV_MODE", "patch")  # "patch"|"cls_patch"
        self.post_seq_mode = getattr(cfg.MODEL.PROMPTSG, "POST_SEQ_MODE", "image")  # "image"|"text_image"|"cls_patch"|"cls_text"|"cls_patch_text"

        # KHÔNG THAY ĐỔI - Giữ nguyên như CLIP-ReID
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024  # CLIP ResNet50 projected global is 1024

        # Classifiers - GIỮ NGUYÊN
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # Bottlenecks - GIỮ NGUYÊN
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # Load CLIP model
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        # Encoders
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        # PromptSG modules
        self.prompt_composer = PromptComposer(clip_model, cfg.MODEL.PROMPTSG.PROMPT_MODE)
        self.inversion = InversionNetwork(dim=512)  # inversion always takes 512-d

        # Projections for RN50 (define in __init__ to ensure optimizer sees params)
        if self.model_name == "RN50":
            self.inversion_projection = nn.Linear(1024, 512)  # for inversion input v
            self.resnet_projection = nn.Linear(1024, 512)     # CLS token for MIM
            self.patch_projection = nn.Linear(1024, 512)      # fallback patch proj if needed
            self.final_projection = nn.Linear(512, 1024)      # for bottleneck_proj input
            self.concat_projection = nn.Linear(512, 1024)     # for inference concat
            self.inversion_projection.apply(weights_init_kaiming)
            self.resnet_projection.apply(weights_init_kaiming)
            self.patch_projection.apply(weights_init_kaiming)
            self.final_projection.apply(weights_init_kaiming)
            self.concat_projection.apply(weights_init_kaiming)

        # MIM (explicit attention + full/eot query + attn_map modes)
        self.mim = MultimodalInteractionModule(
            embed_dim=512,
            num_heads=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS,
            num_blocks=cfg.MODEL.PROMPTSG.POST_CA_BLOCKS,
            mlp_ratio=getattr(cfg.MODEL.PROMPTSG, "MLP_RATIO", 4.0),
            attn_drop=getattr(cfg.MODEL.PROMPTSG, "ATTN_DROPOUT", 0.0),
            proj_drop=getattr(cfg.MODEL.PROMPTSG, "PROJ_DROPOUT", 0.0),
            drop_path=getattr(cfg.MODEL.PROMPTSG, "DROP_PATH", 0.0),
            reweight=self.reweight_mode,
            text_query_mode=self.coattn_text_mode,
            attn_map_mode=self.attn_map_mode,
            attn_pool_mode=getattr(cfg.MODEL.PROMPTSG, "ATTN_POOL_MODE", "mean"),
            eps=getattr(cfg.MODEL.PROMPTSG, "ATTN_EPS", 1e-6),
            kv_mode=self.kv_mode,
            post_seq_mode=self.post_seq_mode,
            act_layer=QuickGELU,
        )

        # Freeze text encoder params (but DO NOT no_grad in forward; gradients flow THROUGH to inversion)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.eval()

        # Cache for simplified prompt
        self._text_cache = None

    def _ensure_text_features(self):
        if self._text_cache is None:
            self.prompt_composer._ensure_embeddings()
            with torch.no_grad():
                prompts = self.prompt_composer.embed_simplified  # (1,L,512)
                tokenized = self.prompt_composer.tokenized_simplified  # (1,L)
                pooled, tokens, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)

            # store on CPU
            self._text_cache = {
                "pooled": pooled.detach().cpu(),     # (1,512)
                "tokens": tokens.detach().cpu(),     # (1,L,512)
                "eot_idx": eot_idx.detach().cpu(),   # (1,)
            }

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None, skip_mim: bool = False):
        """
        Forward pass of PromptSG model
        """
        # Get text features only (pooled)
        if get_text:
            if self.prompt_mode == 'simplified':
                self._ensure_text_features()
                pooled = self._text_cache["pooled"].to(device=x.device).expand(x.shape[0], -1)
                return pooled
            else:
                features_intermediate, features_final, features_proj = self.image_encoder(x)
                if self.model_name == 'ViT-B-16':
                    CLS_proj = features_proj[:, 0]  # (B,512)
                    v = CLS_proj
                else:  # RN50
                    CLS_proj = features_proj[0]     # (B,1024)
                    v = self.inversion_projection(CLS_proj)  # (B,512)

                s_star = self.inversion(v)
                prompts, tokenized = self.prompt_composer(s_star)
                pooled = self.text_encoder(prompts, tokenized)  # (B,512)
                return pooled

        # Get image features only
        if get_image:
            features_intermediate, features_final, features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return features_proj[:, 0]

        # Main forward pass for training/inference
        # For ViT we optionally return extra tensors for the x11-branch (cls_x11, xproj11)
        if self.model_name == 'ViT-B-16':
            features_intermediate, features_final, features_proj, cls_x11, features_proj11 = self.image_encoder(
                x, return_intermediate=True
            )
        else:
            features_intermediate, features_final, features_proj = self.image_encoder(x)
            cls_x11, features_proj11 = None, None

        # Extract features based on backbone type
        # NOTE: during eval we may skip MIM for gallery to save compute
        need_mim = self.training or (not skip_mim)

        if self.model_name == 'ViT-B-16':
            CLS_intermediate = features_intermediate[:, 0]  # (B,768)
            CLS_final = features_final[:, 0]                # (B,768)
            CLS_proj = features_proj[:, 0]                  # (B,512)

            # x11 projected tokens (LN + @proj) for the new triplet branch
            # - cls_x11: (B,768) already returned by ViT (redundant but kept for clarity)
            # - features_proj11: (B,1+M,512)
            if features_proj11 is not None:
                CLS_proj11 = features_proj11[:, 0]           # (B,512)
                patches11 = features_proj11[:, 1:]           # (B,M,512)
                cls_token11 = features_proj11[:, :1]         # (B,1,512)
            else:
                CLS_proj11, patches11, cls_token11 = None, None, None

            if need_mim:
                patches = features_proj[:, 1:]              # (B,M,512)
                cls_token = features_proj[:, :1]            # (B,1,512)
                v = CLS_proj                                # (B,512)

        elif self.model_name == 'RN50':
            CLS_intermediate = F.avg_pool2d(features_intermediate, features_intermediate.shape[2:]).view(x.shape[0], -1)  # (B,2048)
            CLS_final = F.avg_pool2d(features_final, features_final.shape[2:]).view(x.shape[0], -1)                      # (B,2048)
            CLS_proj = features_proj[0]  # (B,1024)

            if need_mim:
                if len(features_proj) > 1:
                    # projected spatial features already 512
                    b, c, h, w = features_proj[1].shape  # c = 512
                    patches = features_proj[1].view(b, c, -1).permute(0, 2, 1)  # (B,M,512)
                    cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)    # (B,1,512)
                else:
                    # fallback: use features_final (1024) -> project to 512
                    b, c, h, w = features_final.shape  # c=1024
                    patches = features_final.view(b, c, -1).permute(0, 2, 1)      # (B,M,1024)
                    patches = self.patch_projection(patches)                      # (B,M,512)
                    cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)     # (B,1,512)

                v = self.inversion_projection(CLS_proj)  # (B,512)

        # Fast path for gallery during evaluation:
        # - gallery: do NOT run inversion/text encoder/MIM; only use visual CLS features
        # - query: goes through full PromptSG pipeline (inversion + text + MIM)
        if (not self.training) and skip_mim:
            if self.neck_feat == 'after':
                feat_gal = self.bottleneck(CLS_final)          # same as query branch (BN on CLS_final)
                feat_proj_gal = self.bottleneck_proj(CLS_proj) # BN on projected CLS (no MIM)
                return torch.cat([feat_gal, feat_proj_gal], dim=1)
            else:
                # 'before': keep raw features (no BN) like original behavior
                CLS_final_gal = CLS_final
                v_final_concat_gal = CLS_proj
                return torch.cat([CLS_final_gal, v_final_concat_gal], dim=1)

        # Generate text features (pooled + full tokens)
        if self.prompt_mode == 'simplified':
            self._ensure_text_features()
            device = x.device if x is not None else next(self.parameters()).device

            pooled = self._text_cache["pooled"].to(device).expand(x.shape[0], -1)           # (B,512)
            tokens = self._text_cache["tokens"].to(device).expand(x.shape[0], -1, -1)      # (B,L,512)

            eot_idx_1 = self._text_cache["eot_idx"].to(device)                              # (1,)
            eot_idx = eot_idx_1.expand(x.shape[0])                                          # (B,)

            text_feat = pooled
            text_tokens_full = tokens

        else:
            s_star = self.inversion(v)  # (B,512)
            prompts, tokenized = self.prompt_composer(s_star)

            # IMPORTANT: no torch.no_grad(), no detach -> allow gradient to flow back to inversion
            text_feat, text_tokens_full, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)        # ========== Multimodal Interaction Module (MIM) ==========
        # Q = full text tokens when COATTN_TEXT_MODE='full'
        sequence, attn_map = self.mim(
            text_tokens_full=text_tokens_full,
            eot_idx=eot_idx,
            patch_tokens=patches,
            cls_token=cls_token,
            return_cls_states=False
        )

        # Pool to a single vector (use CLIP EOT/EOS position when query is full tokens)
        if sequence.dim() == 3:
            B = sequence.size(0)
            if sequence.size(1) == 1:
                v_final = sequence[:, 0, :]  # (B,512)
            else:
                v_final = sequence[torch.arange(B, device=sequence.device), eot_idx, :]  # (B,512)
        else:
            v_final = sequence  # (B,512)

        # ===== New: x11-branch MIM feature for Triplet (replace CLS_intermediate) =====
        # - Use CLS_proj11 (x11 projected to 512) -> inversion -> composed prompt -> text encoder
        # - Run MIM with x11's (cls/patch) tokens to get a 512-D feature
        # - Only used for training triplet; SupCon + ID remain unchanged.
        v_final_11 = CLS_intermediate
        if self.training and (self.model_name == 'ViT-B-16') and (CLS_proj11 is not None) and (cls_token11 is not None):
            if self.prompt_mode == 'simplified':
                # simplified mode has no s*: reuse cached text tokens
                text_tokens_full_11 = text_tokens_full
                eot_idx_11 = eot_idx
            else:
                s_star_11 = self.inversion(CLS_proj11)  # (B,512)
                prompts_11, tokenized_11 = self.prompt_composer(s_star_11)
                _, text_tokens_full_11, eot_idx_11 = self.text_encoder(prompts_11, tokenized_11, return_tokens=True)

            seq11, _ = self.mim(
                text_tokens_full=text_tokens_full_11,
                eot_idx=eot_idx_11,
                patch_tokens=patches11,
                cls_token=cls_token11,
                return_cls_states=False,
            )

            # pool seq11 -> (B,512)
            if seq11.dim() == 3:
                B11 = seq11.size(0)
                if seq11.size(1) == 1:
                    v_final_11 = seq11[:, 0, :]
                else:
                    v_final_11 = seq11[torch.arange(B11, device=seq11.device), eot_idx_11, :]
            else:
                v_final_11 = seq11

        # Debug: print shapes once
        if not hasattr(self, "_logged_mim_seq_shape"):
            print(f"[MIM] sequence shape={tuple(sequence.shape)} | v_final(EOT) shape={tuple(v_final.shape)}")
            self._logged_mim_seq_shape = True

        # ========== Bottleneck Layers ==========
        feat = self.bottleneck(CLS_final)  # (B,768/2048)

        if self.model_name == 'RN50':
            feat_proj_input = self.final_projection(v_final)  # (B,1024)
        else:
            feat_proj_input = v_final  # (B,512)

        feat_proj = self.bottleneck_proj(feat_proj_input)

        # ========== Output ==========
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)

            # multi-scale triplet features like your original
            # Replace CLS_intermediate with x11-branch MIM output
            triplet_feats = [v_final_11, CLS_final, v_final]

            return [cls_score, cls_score_proj], triplet_feats, v, text_feat

        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                if self.model_name == 'RN50':
                    v_final_concat = self.concat_projection(v_final)  # (B,1024)
                else:
                    v_final_concat = v_final  # (B,512)
                return torch.cat([CLS_final, v_final_concat], dim=1)

    def load_param(self, trained_path):
        """Load pretrained parameters"""
        self.prompt_composer._ensure_embeddings()

        param_dict = torch.load(trained_path, map_location='cpu')
        for key in param_dict:
            new_key = key.replace('module.', '')
            if new_key in self.state_dict():
                if self.state_dict()[new_key].shape == param_dict[key].shape:
                    self.state_dict()[new_key].copy_(param_dict[key])
                else:
                    print(f"Skipping {new_key}: shape mismatch {self.state_dict()[new_key].shape} vs {param_dict[key].shape}")

    def forward_with_attention(self, x):
        """
        Forward pass returning attention maps for visualization (GradCAM-like).
        NOTE: this is typically used in eval/visualization; using no_grad for text is fine.
        """
        features_intermediate, features_final, features_proj = self.image_encoder(x)

        if self.model_name == 'ViT-B-16':
            CLS_intermediate = features_intermediate[:, 0]
            CLS_final = features_final[:, 0]
            CLS_proj = features_proj[:, 0]
            patches = features_proj[:, 1:]
            cls_token = features_proj[:, :1]
            v = CLS_proj

        elif self.model_name == 'RN50':
            CLS_intermediate = F.avg_pool2d(features_intermediate, features_intermediate.shape[2:]).view(x.shape[0], -1)
            CLS_final = F.avg_pool2d(features_final, features_final.shape[2:]).view(x.shape[0], -1)
            CLS_proj = features_proj[0]
            v = self.inversion_projection(CLS_proj)

            if len(features_proj) > 1:
                b, c, h, w = features_proj[1].shape
                patches = features_proj[1].view(b, c, -1).permute(0, 2, 1)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)
            else:
                b, c, h, w = features_final.shape
                patches = features_final.view(b, c, -1).permute(0, 2, 1)
                patches = self.patch_projection(patches)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)

        # text
        if self.prompt_mode == 'simplified':
            self._ensure_text_features()
            device = x.device
            text_feat = self._text_cache["pooled"].to(device).expand(x.shape[0], -1)
            text_tokens_full = self._text_cache["tokens"].to(device).expand(x.shape[0], -1, -1)
            eot_idx = self._text_cache["eot_idx"].to(device).expand(x.shape[0])
        else:
            s_star = self.inversion(v)
            prompts, tokenized = self.prompt_composer(s_star)
            text_feat, text_tokens_full, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)

        # MIM
        sequence, attn_map, cls_states = self.mim(
            text_tokens_full=text_tokens_full,
            eot_idx=eot_idx,
            patch_tokens=patches,
            cls_token=cls_token,
            return_cls_states=True
        )
        v_final = cls_states[-1]

        # bottleneck + cls
        feat = self.bottleneck(CLS_final)
        if self.model_name == 'RN50':
            feat_proj_input = self.final_projection(v_final)
        else:
            feat_proj_input = v_final
        feat_proj = self.bottleneck_proj(feat_proj_input)

        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)

        return {
            'logits': [cls_score, cls_score_proj],
            'features': torch.cat([feat, feat_proj], dim=1),
            'mim_attention': attn_map,     # (B,1,num_patches)
            'patch_tokens': patches,
            'text_features': text_feat,
            'text_tokens_full': text_tokens_full,
            'eot_idx': eot_idx,
            'cls_states': cls_states,
            'v_final': v_final
        }

    def get_attention_map(self, x, reshape_to_image=True):
        result = self.forward_with_attention(x)
        attn_map = result['mim_attention'].squeeze(1)  # (B, num_patches)

        if reshape_to_image:
            B, N = attn_map.shape
            attn_map = attn_map.view(B, self.h_resolution, self.w_resolution)

        return attn_map


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num, view_num):
    method = getattr(cfg.MODEL, "METHOD", "promptsg")
    if method == "clip_scgi":
        return CLIPSCGIModel(num_class, camera_num, view_num, cfg)
    return PromptSGModel(num_class, camera_num, view_num, cfg)

