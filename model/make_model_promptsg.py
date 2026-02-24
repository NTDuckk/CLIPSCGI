import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


# -------------------------
# Init helpers (keep)
# -------------------------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class LayerNorm(nn.LayerNorm):
    """fp16-safe LayerNorm (CLIP-style)"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# -------------------------
# CLIP text encoder wrapper
# -------------------------
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
        x = x.permute(1, 0, 2)  # (L, B, C)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, L, C)
        x = self.ln_final(x).type(self.dtype)

        tokens_proj = x @ self.text_projection  # (B, L, 512)
        eot_idx = tokenized_prompts.argmax(dim=-1)  # (B,)
        pooled = tokens_proj[
            torch.arange(tokens_proj.size(0), device=tokens_proj.device),
            eot_idx
        ]  # (B,512)

        if return_tokens:
            return pooled, tokens_proj, eot_idx
        return pooled


# -------------------------
# Prompt composer for:
#  - "a photo of a X person"  (composed)
#  - "a photo of a person"    (simplified)
# -------------------------
class PromptComposer(nn.Module):
    def __init__(self, clip_model, mode: str):
        super().__init__()
        assert mode in ["composed", "simplified"]
        self.mode = mode
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        # paper strings (case-insensitive for tokenizer, ok)
        self.composed_str = "a photo of a X person"
        self.simplified_str = "a photo of a person"

        self.register_buffer("tokenized_composed", torch.empty(0, dtype=torch.long))
        self.register_buffer("tokenized_simplified", torch.empty(0, dtype=torch.long))
        self.register_buffer("embed_composed", torch.empty(0))
        self.register_buffer("embed_simplified", torch.empty(0))
        self.x_pos = None

    def _ensure_tokenization(self):
        if self.tokenized_composed.numel() != 0:
            return

        import model.clip.clip as clip_module
        dev = self.token_embedding.weight.device

        tokenized_composed = clip_module.tokenize([self.composed_str]).to(dev)
        tokenized_simplified = clip_module.tokenize([self.simplified_str]).to(dev)

        tokenized_x = clip_module.tokenize(["X"]).to(dev)
        x_token_id = tokenized_x[0, 1].item()

        x_pos = (tokenized_composed[0] == x_token_id).nonzero(as_tuple=False)
        if x_pos.numel() == 0:
            raise ValueError("Cannot locate placeholder token 'X' in composed prompt")

        self.tokenized_composed = tokenized_composed
        self.tokenized_simplified = tokenized_simplified
        self.x_pos = int(x_pos[0].item())

    def _ensure_embeddings(self):
        self._ensure_tokenization()
        if self.embed_composed.numel() != 0:
            return
        with torch.no_grad():
            self.embed_composed = self.token_embedding(self.tokenized_composed).type(self.dtype)       # (1,L,C)
            self.embed_simplified = self.token_embedding(self.tokenized_simplified).type(self.dtype)   # (1,L,C)

    def forward(self, s_star: torch.Tensor = None, batch_size: int = None):
        """
        Returns:
          prompts: (B, L, C)
          tokenized: (B, L)
        """
        self._ensure_embeddings()

        if batch_size is None:
            if s_star is None:
                raise ValueError("Provide either s_star or batch_size")
            batch_size = s_star.size(0)

        if self.mode == "simplified":
            tokenized = self.tokenized_simplified.expand(batch_size, -1)
            prompts = self.embed_simplified.expand(batch_size, -1, -1)
            return prompts, tokenized

        # composed mode needs s_star
        if s_star is None:
            raise ValueError("composed prompt needs s_star")

        s_star = s_star.to(dtype=self.embed_composed.dtype)
        tokenized = self.tokenized_composed.expand(batch_size, -1)

        prefix = self.embed_composed[:, :self.x_pos, :].expand(batch_size, -1, -1)
        suffix = self.embed_composed[:, self.x_pos + 1 :, :].expand(batch_size, -1, -1)
        prompts = torch.cat([prefix, s_star.unsqueeze(1), suffix], dim=1)
        return prompts, tokenized


# -------------------------
# CGI local query block (Eq.4-5)
#   QC = Q + CrossAttn(LN(Q), LN(ṽi), LN(ṽi))
#   P  = QC + FFN(QC)
# -------------------------
class CGIQueryBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=QuickGELU,
    ):
        super().__init__()
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )

        self.norm_ffn = LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        # q: (B,K,D), kv: (B,M,D)
        q_ln = self.norm_q(q)
        kv_ln = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + self.drop_path(attn_out)                  # Eq.(4)
        q = q + self.drop_path(self.ffn(self.norm_ffn(q)))  # Eq.(5)
        return q


def _mlp_3layer(dim: int):
    # paper: fMg/fMl are 3-layer FC networks
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim),
    )


class CaptionGuidedInversion(nn.Module):
    """
    CLIP-SCGI CGI module:
      ṽ = v ⊙ t                                   (Eq.3)
      S*_global = fMg(ṽ_cls)                      (paper)
      Q -> blocks: QC = Q + CrossAttn(...), P=QC+FFN  (Eq.4-5)
      S*_local  = fMl(Avg(P))                      (paper)
      S* = S*_global + S*_local                    (paper)
    """
    def __init__(
        self,
        embed_dim: int = 512,
        num_queries: int = 2,
        num_heads: int = 8,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            CGIQueryBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path,
            )
            for _ in range(int(depth))
        ])

        self.fMg = _mlp_3layer(embed_dim)
        self.fMl = _mlp_3layer(embed_dim)

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor, cap_feat: torch.Tensor):
        """
        Args:
          cls_token:    (B,512)
          patch_tokens: (B,M,512)
          cap_feat:     (B,512)  (caption pooled embedding)
        Returns:
          s_star:   (B,512)
          cls_ref:  (B,512)  refined cls
          patch_ref:(B,M,512) refined patches
        """
        # Eq.(3): refine visual tokens with caption embedding
        cls_ref = cls_token * cap_feat
        patch_ref = patch_tokens * cap_feat.unsqueeze(1)

        # local: learnable queries attend to refined patches via CGI blocks
        B = cls_token.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B,K,512)
        for blk in self.blocks:
            q = blk(q, patch_ref)  # (B,K,512)
        local_pool = q.mean(dim=1)  # Avg(P) -> (B,512)

        s_global = self.fMg(cls_ref)      # (B,512)
        s_local = self.fMl(local_pool)    # (B,512)
        s_star = s_global + s_local       # paper

        return s_star, cls_ref, patch_ref


# -------------------------
# CFF module (cross-attn + 2 transformer blocks)
# -------------------------
class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=QuickGELU,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)

        self.norm2 = LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_ln = self.norm1(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ContextualFeatureFusion(nn.Module):
    """
    paper: multi-head cross-attn + 2 transformer blocks
      - Q: text inversion tokens {t̂_sos...t̂_eos}
      - K/V: image tokens (you can choose cls / cls_patch via kv_mode)
    """
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 2,
        kv_mode: str = "cls_patch",  # "cls" | "patch" | "cls_patch"
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert kv_mode in ["cls", "patch", "cls_patch"]
        self.kv_mode = kv_mode

        self.norm_q = LayerNorm(embed_dim)
        self.norm_kv = LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)

        self.post_blocks = nn.ModuleList([
            SelfAttnBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path,
            )
            for _ in range(int(num_blocks))
        ])

    def forward(self, text_tokens: torch.Tensor, cls_token_1: torch.Tensor, patch_tokens: torch.Tensor):
        """
        Args:
          text_tokens: (B,L,512)  query sequence
          cls_token_1: (B,1,512)
          patch_tokens:(B,M,512)
        Returns:
          seq: (B,L,512)
        """
        q = self.norm_q(text_tokens)

        if self.kv_mode == "cls":
            kv = cls_token_1
        elif self.kv_mode == "patch":
            kv = patch_tokens
        else:
            kv = torch.cat([cls_token_1, patch_tokens], dim=1)
        kv = self.norm_kv(kv)

        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        seq = text_tokens + out  # residual on query stream

        for blk in self.post_blocks:
            seq = blk(seq)
        return seq

# -------------------------
# CLIP-SCGI main model
# -------------------------
class CLIPSCGIModel(nn.Module):
    """
    CLIP-SCGI:
      - offline captions -> cap_feat
      - CGI -> s* and refined tokens
      - composed prompt "a photo of a S* person" -> text tokens
      - CFF fuses text tokens with image tokens
      - losses computed in processor: L_id + L_tri + L_con
    Inference:
      - drop CGI, fixed prompt "a photo of a person"
      - still run CFF (but text tokens can be cached since prompt fixed)
    """
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = num_classes

        # ViT projected dim = 512 in your codebase
        self.in_planes = 512

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # Load CLIP
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        # prompt composers
        self.prompt_composed = PromptComposer(clip_model, mode="composed")
        self.prompt_fixed = PromptComposer(clip_model, mode="simplified")

        # SCGI cfg
        scgi = getattr(cfg.MODEL, "CLIPSCGI", None)

        cgi_num_q = getattr(scgi, "CGI_NUM_QUERIES", 2) if scgi is not None else 2
        cgi_heads = getattr(scgi, "CGI_HEADS", 8) if scgi is not None else 8
        cgi_depth = getattr(scgi, "CGI_DEPTH", 1) if scgi is not None else 1

        cff_heads = getattr(scgi, "CFF_HEADS", 8) if scgi is not None else 8
        cff_blocks = getattr(scgi, "CFF_BLOCKS", 2) if scgi is not None else 2
        kv_mode = getattr(scgi, "KV_MODE", "cls_patch") if scgi is not None else "cls_patch"

        self.cgi = CaptionGuidedInversion(
            embed_dim=512,
            num_queries=cgi_num_q,
            num_heads=cgi_heads,
            depth=cgi_depth,
        )

        self.cff = ContextualFeatureFusion(
            embed_dim=512,
            num_heads=cff_heads,
            num_blocks=cff_blocks,
            kv_mode=kv_mode,
        )

        self.drop_cgi_in_infer = getattr(scgi, "DROP_CGI_IN_INFER", True) if scgi is not None else True
        self.caption_no_grad = getattr(scgi, "CAPTION_NO_GRAD", True) if scgi is not None else True

        # cache fixed prompt tokens for inference (optional)
        self._fixed_cache = None

    def _encode_caption_pooled(self, captions, device):
        """
        cap_feat = f_txt(Tk) pooled (EOT), used as 't' in Eq.(3) ṽ=v⊙t.
        Default: no_grad for caption branch (treat caption as fixed guidance).
        """
        import model.clip.clip as clip_module
        tokenized = clip_module.tokenize(list(captions)).to(device)
        prompts = self.token_embedding(tokenized).type(self.dtype)

        if self.caption_no_grad:
            with torch.no_grad():
                cap_feat = self.text_encoder(prompts, tokenized)  # (B,512)
        else:
            cap_feat = self.text_encoder(prompts, tokenized)
        return cap_feat

    def _get_fixed_prompt_tokens(self, B, device, dtype):
        # cache only for eval (fixed prompt)
        if self._fixed_cache is None:
            with torch.no_grad():
                prompts, tokenized = self.prompt_fixed(s_star=None, batch_size=1)
                pooled, tokens, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)
            self._fixed_cache = {
                "prompts": prompts.detach().cpu(),
                "tokenized": tokenized.detach().cpu(),
                "tokens": tokens.detach().cpu(),
                "eot_idx": eot_idx.detach().cpu(),
            }

        tokens = self._fixed_cache["tokens"].to(device=device, dtype=dtype).expand(B, -1, -1)
        eot_idx = self._fixed_cache["eot_idx"].to(device=device).expand(B)
        return tokens, eot_idx

    def forward(self, x=None, label=None, caption=None, img_paths=None, skip_mim: bool = False, **kwargs):
        """
        Return signature (compatible processor):
          cls_score, feat_for_triplet, image_contrast_feat, text_contrast_feat
        """
        if self.model_name != "ViT-B-16":
            raise NotImplementedError("This SCGI implementation currently assumes ViT-B-16 tokens output (B,1+M,512).")

        # Your visual returns (_, _, xproj) where xproj is token sequence (B,1+M,512)
        _, _, xproj = self.image_encoder(x)
        cls_token = xproj[:, 0]       # (B,512)
        patch_tokens = xproj[:, 1:]   # (B,M,512)

        device = xproj.device
        B = xproj.size(0)

        # -------------------------
        # Inference: drop CGI, fixed prompt
        # -------------------------
        if (not self.training) and self.drop_cgi_in_infer:
            if self._fixed_cache is None:
                # make sure token embeddings exist
                self.prompt_fixed._ensure_embeddings()

            text_tokens, eot_idx = self._get_fixed_prompt_tokens(B, device=device, dtype=xproj.dtype)

            if skip_mim:
                feat = cls_token
            else:
                seq = self.cff(
                    text_tokens=text_tokens,
                    cls_token_1=cls_token.unsqueeze(1),
                    patch_tokens=patch_tokens,
                )
                feat = seq[torch.arange(B, device=device), eot_idx, :]  # EOT pooled

            bn_feat = self.bottleneck(feat)
            return bn_feat if self.neck_feat == "after" else feat

        # -------------------------
        # Training: need caption -> CGI -> composed prompt -> CFF
        # -------------------------
        if caption is None:
            # fallback: behave like fixed prompt (still runs CFF)
            text_tokens, eot_idx = self._get_fixed_prompt_tokens(B, device=device, dtype=xproj.dtype)
            cls_ref, patch_ref = cls_token, patch_tokens
            pooled_prompt = None
        else:
            if isinstance(caption, (list, tuple)):
                caption = [c if c is not None else "" for c in caption]

            cap_feat = self._encode_caption_pooled(caption, device=device)  # (B,512)

            # CGI -> s* and refined tokens
            s_star, cls_ref, patch_ref = self.cgi(cls_token, patch_tokens, cap_feat)

            # composed prompt "a photo of a S* person"
            prompts, tokenized = self.prompt_composed(s_star=s_star, batch_size=B)
            pooled_prompt, text_tokens, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)

        # CFF fusion
        if skip_mim:
            feat = cls_ref
        else:
            seq = self.cff(
                text_tokens=text_tokens,
                cls_token_1=cls_ref.unsqueeze(1),
                patch_tokens=patch_ref,
            )
            feat = seq[torch.arange(B, device=device), eot_idx, :]  # EOT pooled

        bn_feat = self.bottleneck(feat)
        cls_score = self.classifier(bn_feat)

        # Contrastive pair for L_con should be:
        #  - image: (refined) cls token
        #  - text:  pooled embedding of composed prompt (a photo of a S* person)
        image_contrast_feat = cls_ref
        text_contrast_feat = pooled_prompt  # (B,512) or None if fallback

        return cls_score, feat, image_contrast_feat, text_contrast_feat


# -------------------------
# CLIP loader (keep same as your codebase)
# -------------------------
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
    # only SCGI in this file
    return CLIPSCGIModel(num_class, camera_num, view_num, cfg)