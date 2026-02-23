import logging
import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist
from torch.nn import functional as F
import subprocess
import sys

# ====== ADDED: attention visualization helpers ======
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

# CLIP default norm (fallback)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _dist_is_main():
    """True if not distributed, or rank==0."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _unwrap_model(model):
    """Handle DataParallel/DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def _get_norm_stats(cfg, device):
    """
    Try to read cfg.INPUT.PIXEL_MEAN/PIXEL_STD (if exists).
    Fallback to CLIP mean/std.
    """
    mean = None
    std = None
    try:
        mean = getattr(getattr(cfg, "INPUT", object()), "PIXEL_MEAN", None)
        std  = getattr(getattr(cfg, "INPUT", object()), "PIXEL_STD", None)
    except Exception:
        mean, std = None, None

    if mean is None or std is None:
        mean, std = _CLIP_MEAN, _CLIP_STD

    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    std  = torch.tensor(std, device=device).view(3, 1, 1)
    return mean, std


@torch.no_grad()
def _save_attention_triplet(cfg, model, imgs, output_dir, epoch_idx, iter_idx, k_idx=0, logger=None):
    """
    Save one random sample from this batch as a 1x3 figure:
      [original | attention | overlay]
    Requires model to implement forward_with_attention() and h_resolution/w_resolution.
    """
    if not _dist_is_main():
        return

    m = _unwrap_model(model)

    if not hasattr(m, "forward_with_attention"):
        if logger:
            logger.warning("[ATTN] model has no forward_with_attention(); skipping attention visualization.")
        return

    # pick 1 sample in batch
    B = imgs.size(0)
    j = random.randrange(B)

    # temporarily eval so extra forward won't update BN stats
    was_training = m.training
    m.eval()

    try:
        with amp.autocast(enabled=True):
            out = m.forward_with_attention(imgs)  # dict with 'mim_attention'
    except Exception as e:
        if logger:
            logger.warning(f"[ATTN] forward_with_attention failed: {e}")
        if was_training:
            m.train()
        return

    if was_training:
        m.train()

    if "mim_attention" not in out:
        if logger:
            logger.warning("[ATTN] forward_with_attention() output missing 'mim_attention'; skipping.")
        return

    attn = out["mim_attention"]  # (B,1,M)
    if attn.dim() != 3 or attn.size(1) != 1:
        if logger:
            logger.warning(f"[ATTN] unexpected attn shape={tuple(attn.shape)}; expected (B,1,M).")
        return

    # get patch grid size
    M = attn.size(-1)
    hr = getattr(m, "h_resolution", None)
    wr = getattr(m, "w_resolution", None)
    if hr is None or wr is None or hr * wr != M:
        s = int(round(M ** 0.5))
        hr, wr = s, s

    # reshape -> upsample to image size
    attn_hw = attn[j, 0].view(1, 1, hr, wr)  # (1,1,hr,wr)
    H, W = imgs.size(-2), imgs.size(-1)
    attn_up = F.interpolate(attn_hw, size=(H, W), mode="bilinear", align_corners=False)[0, 0]  # (H,W)

    # normalize attn to [0,1]
    attn_up = attn_up - attn_up.min()
    attn_up = attn_up / (attn_up.max() + 1e-6)
    attn_np = attn_up.detach().float().cpu().numpy()

    # denormalize image to [0,1]
    mean, std = _get_norm_stats(cfg, device=imgs.device)
    img_dn = (imgs[j] * std + mean).clamp(0, 1)  # (3,H,W)
    img_np = img_dn.detach().float().cpu().permute(1, 2, 0).numpy()  # (H,W,3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ep{epoch_idx:03d}_it{iter_idx:05d}_k{k_idx}.png")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img_np); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(attn_np, cmap="jet"); ax[1].set_title("Attention"); ax[1].axis("off")
    ax[2].imshow(img_np); ax[2].imshow(attn_np, cmap="jet", alpha=0.45)
    ax[2].set_title("Overlay"); ax[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"[ATTN] saved: {save_path}")
# ====== END ADDED ======


def setup_training_logger(cfg):
    """Setup additional file logger for training metrics"""
    log_dir = cfg.OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)

    metrics_logger = logging.getLogger("promptsg.metrics")
    metrics_logger.setLevel(logging.INFO)

    metrics_file = os.path.join(log_dir, 'training_metrics.txt')
    file_handler = logging.FileHandler(metrics_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not metrics_logger.handlers:
        metrics_logger.addHandler(file_handler)

    return metrics_logger


def auto_generate_plots(cfg):
    """Automatically generate learning curves after training completion"""
    logger = logging.getLogger("promptsg.train")
    logger.info("Generating learning curves...")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        plot_script = os.path.join(project_root, 'plot_learning_curves.py')

        if not os.path.exists(plot_script):
            logger.warning(f"Plot script not found at {plot_script}")
            return False

        cmd = [
            sys.executable,
            plot_script,
            '--log_dir', cfg.OUTPUT_DIR,
            '--output_dir', os.path.join(project_root, 'plots'),
            '--save_json'
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.info("Learning curves generated successfully!")
            logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.info(f"Warnings: {result.stderr}")
            return True
        else:
            logger.error(f"Failed to generate learning curves: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Plot generation timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        return False


def do_train(cfg, model, train_loader, val_loader, query_loader, gallery_loader,
             optimizer, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.PROMPTSG.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.PROMPTSG.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.PROMPTSG.EVAL_PERIOD
    epochs = cfg.SOLVER.PROMPTSG.MAX_EPOCHS

    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("promptsg.train")
    logger.info('start training')
    logger.info("Config:\n{}".format(cfg.dump()))

    metrics_logger = setup_training_logger(cfg)
    metrics_logger.info("=== TRAINING STARTED ===")
    metrics_logger.info(f"Model: {cfg.MODEL.NAME}")
    # PromptSG key may not exist for SCGI, so guard:
    prompt_mode = getattr(getattr(cfg.MODEL, "PROMPTSG", object()), "PROMPT_MODE", "N/A")
    metrics_logger.info(f"Prompt mode: {prompt_mode}")
    metrics_logger.info(f"Max epochs: {epochs}")
    metrics_logger.info(f"Learning rate: {cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_logger.info(f"Total parameters: {total_params:,}")
    metrics_logger.info(f"Trainable parameters: {trainable_params:,}")
    metrics_logger.info("="*50)

    if device:
        device = torch.device(f"cuda:{local_rank}") if local_rank is not None else torch.device("cuda")
        model.to(device)
        # IMPORTANT: don't wrap DataParallel when using distributed training
        if (not getattr(cfg.MODEL, "DIST_TRAIN", False)) and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    id_meter = AverageMeter()
    tri_meter = AverageMeter()
    supcon_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset(); acc_meter.reset(); evaluator.reset()
        id_meter.reset(); tri_meter.reset(); supcon_meter.reset()

        logger.info("Epoch {} started".format(epoch))
        metrics_logger.info(f"EPOCH {epoch} - Training started")
        model.train()

        # ====== ADDED: pick 5 random iterations each epoch to save attention triplets ======
        num_iters = len(train_loader)
        k_vis = 5
        vis_iters = set(random.sample(range(num_iters), k=min(k_vis, num_iters))) if num_iters > 0 else set()
        vis_count = 0
        epoch_vis_dir = os.path.join(cfg.OUTPUT_DIR, "attn_vis", f"epoch_{epoch:03d}")
        if _dist_is_main():
            os.makedirs(epoch_vis_dir, exist_ok=True)
            logger.info(f"[ATTN] will save {min(k_vis, num_iters)} samples this epoch to: {epoch_vis_dir}")
        # ====== END ADDED ======

        for n_iter, batch in enumerate(train_loader):
            # PromptSG batch: (img, pid, camid, viewid)
            # CLIP-SCGI batch: (img, pid, camid, viewid, captions)
            if len(batch) == 5:
                img, pid, camid, viewid, captions = batch
            else:
                img, pid, camid, viewid = batch
                captions = None

            img = img.to(device, non_blocking=True)
            target = pid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=True):
                method = getattr(cfg.MODEL, 'METHOD', 'promptsg')
                if method == 'clip_scgi':
                    cls_score, triplet_feats, image_feat, text_feat = model(x=img, label=target, caption=captions)
                else:
                    cls_score, triplet_feats, image_feat, text_feat = model(x=img, label=target)
                total_loss, losses_dict = loss_fn(cls_score, triplet_feats, target, camid, image_feat, text_feat)

            scaler.scale(total_loss).backward()

            # ---- gradient norms (unwrap for DP/DDP) ----
            m = _unwrap_model(model)

            def _grad_norm(module) -> float:
                if module is None:
                    return 0.0
                s = 0.0
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    s += float(g.norm().item()) ** 2
                return s ** 0.5

            inv_grad_norm = _grad_norm(getattr(m, 'inversion', None))
            mim_grad_norm = _grad_norm(getattr(m, 'mim', None))
            vis_grad_norm = _grad_norm(getattr(m, 'image_encoder', None))
            text_grad_norm = _grad_norm(getattr(m, 'text_encoder', None))

            scaler.step(optimizer)
            scaler.update()

            # ====== ADDED: save random 5x per epoch (image | attn | overlay) ======
            if n_iter in vis_iters:
                _save_attention_triplet(
                    cfg=cfg,
                    model=model,
                    imgs=img,
                    output_dir=epoch_vis_dir,
                    epoch_idx=epoch,
                    iter_idx=n_iter,
                    k_idx=vis_count,
                    logger=logger
                )
                vis_count += 1
            # ====== END ADDED ======

            # ---- meters/logging (FIXED) ----
            with torch.no_grad():
                main_cls_score = cls_score[0] if isinstance(cls_score, (list, tuple)) else cls_score
                acc = (main_cls_score.max(1)[1] == target).float().mean()

                loss_meter.update(float(total_loss.item()), img.shape[0])
                id_meter.update(float(losses_dict['id_loss'].item()), img.shape[0])
                tri_meter.update(float(losses_dict['tri_loss'].item()), img.shape[0])

                sup_or_con = losses_dict.get('supcon_loss', losses_dict.get('con_loss', torch.tensor(0.0)))
                try:
                    supcon_val = float(sup_or_con.item())
                except Exception:
                    supcon_val = 0.0

                supcon_meter.update(supcon_val, img.shape[0])
                acc_meter.update(float(acc.item()), img.shape[0])

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                lr0 = optimizer.param_groups[0]['lr']
                log_msg = (
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (ID {:.3f} TRI {:.3f} SupCon {:.3f}) "
                    "Acc: {:.3f} Lr: {:.2e} Grad: Inv {:.4f} MIM {:.4f} Vis {:.4f} Text {:.4f}"
                ).format(
                    epoch, n_iter + 1, len(train_loader),
                    loss_meter.avg, id_meter.avg, tri_meter.avg, supcon_meter.avg,
                    acc_meter.avg,
                    lr0,
                    inv_grad_norm, mim_grad_norm, vis_grad_norm, text_grad_norm
                )
                logger.info(log_msg)
                metrics_logger.info(log_msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if not getattr(cfg.MODEL, "DIST_TRAIN", False):
            epoch_msg = "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                epoch, time_per_batch, train_loader.batch_size / time_per_batch
            )
            logger.info(epoch_msg)
            metrics_logger.info(epoch_msg)

        # Step scheduler AFTER epoch
        scheduler.step()

        # Save checkpoint
        if epoch % checkpoint_period == 0:
            if getattr(cfg.MODEL, "DIST_TRAIN", False):
                if dist.get_rank() == 0:
                    torch.save(_unwrap_model(model).state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(_unwrap_model(model).state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # ---- Evaluation ----
        if epoch % eval_period == 0:
            method = getattr(cfg.MODEL, 'METHOD', 'promptsg')
            skip_gallery = getattr(getattr(cfg, 'TEST', object()), 'SKIP_MIM_GALLERY', False)
            if method == 'clip_scgi':
                skip_gallery = False  # SCGI: use CFF for both query+gallery

            if getattr(cfg.MODEL, "DIST_TRAIN", False):
                if dist.get_rank() == 0:
                    model.eval()
                    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                    evaluator.reset()

                    # query (with MIM/CFF)
                    for _, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device, non_blocking=True)
                            feat = model(img, skip_mim=False)
                            evaluator.update((feat, pid, camid))

                    # gallery
                    for _, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(gallery_loader):
                        with torch.no_grad():
                            img = img.to(device, non_blocking=True)
                            feat = model(img, skip_mim=skip_gallery)
                            evaluator.update((feat, pid, camid))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    val_msg = "Validation Results - Epoch {}".format(epoch)
                    logger.info(val_msg); metrics_logger.info(val_msg)
                    map_msg = "mAP: {:.1%}".format(mAP)
                    logger.info(map_msg); metrics_logger.info(map_msg)
                    for r in [1, 5, 10]:
                        rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                        logger.info(rank_msg); metrics_logger.info(rank_msg)
                    torch.cuda.empty_cache()
            else:
                model.eval()
                evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                evaluator.reset()

                # query (with MIM/CFF)
                for _, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device, non_blocking=True)
                        feat = model(img, skip_mim=False)
                        evaluator.update((feat, pid, camid))

                # gallery (FIXED: NOT nested inside query loop)
                for _, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device, non_blocking=True)
                        feat = model(img, skip_mim=skip_gallery)
                        evaluator.update((feat, pid, camid))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                val_msg = "Validation Results - Epoch {}".format(epoch)
                logger.info(val_msg); metrics_logger.info(val_msg)
                map_msg = "mAP: {:.1%}".format(mAP)
                logger.info(map_msg); metrics_logger.info(map_msg)
                for r in [1, 5, 10]:
                    rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                    logger.info(rank_msg); metrics_logger.info(rank_msg)
                torch.cuda.empty_cache()

    total_time = time.monotonic() - all_start_time
    time_msg = "Total running time: {:.1f}[s]".format(total_time)
    logger.info(time_msg)
    metrics_logger.info("="*50)
    metrics_logger.info("=== TRAINING COMPLETED ===")
    metrics_logger.info(time_msg)
    metrics_logger.info("="*50)

    if (not getattr(cfg.MODEL, "DIST_TRAIN", False)) or (getattr(cfg.MODEL, "DIST_TRAIN", False) and dist.get_rank() == 0):
        logger.info("Training completed. Auto-generating learning curves...")
        success = auto_generate_plots(cfg)
        if success:
            logger.info("Training completed successfully! Check 'plots/' directory for learning curves.")
        else:
            logger.warning("Plot generation failed. You can manually run: python plot_learning_curves.py")
    else:
        logger.info("Training completed. Plot generation skipped for distributed training.")

    return


def do_inference(cfg, model, val_loader, num_query):
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("promptsg.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        device = torch.device("cuda") if device == "cuda" or str(device).startswith("cuda") else torch.device(device)
        # Avoid DataParallel in distributed setting
        if (not getattr(cfg.MODEL, "DIST_TRAIN", False)) and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    # Decide gallery behavior once
    method = getattr(cfg.MODEL, 'METHOD', 'promptsg')
    skip_gallery = getattr(getattr(cfg, 'TEST', object()), 'SKIP_MIM_GALLERY', False)
    if method == 'clip_scgi':
        skip_gallery = False

    # Split query/gallery inside each batch because val_loader is [query + gallery]
    n_seen = 0
    for _, (img, pid, camid, camid_batch, viewid, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device, non_blocking=True)
            B = img.size(0)

            q_remain = max(0, num_query - n_seen)
            q_bsz = min(B, q_remain)

            feats = []
            if q_bsz > 0:
                feats.append(model(img[:q_bsz], skip_mim=False))  # query: with MIM/CFF
            if q_bsz < B:
                feats.append(model(img[q_bsz:], skip_mim=skip_gallery))  # gallery: optional skip (PromptSG); never skip (SCGI)

            feat = torch.cat(feats, dim=0) if len(feats) > 1 else feats[0]
            evaluator.update((feat, pid, camid))
            n_seen += B

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4], mAP