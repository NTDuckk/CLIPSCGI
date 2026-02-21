import torch


def make_optimizer(cfg, model):
    params_visual = []
    params_new = []

    for name, p in model.named_parameters():
        # Exclude text encoder and prompt composer like CLIP-ReID
        if "text_encoder" in name:
            p.requires_grad_(False)
            continue
        if "prompt_composer" in name:
            p.requires_grad_(False)
            continue
        if not p.requires_grad:
            continue
        if name.startswith('image_encoder.'):
            params_visual.append(p)
        else:
            params_new.append(p)

    optim_name = cfg.SOLVER.PROMPTSG.OPTIMIZER_NAME
    wd = cfg.SOLVER.PROMPTSG.WEIGHT_DECAY

    if optim_name == 'SGD':
        optimizer = getattr(torch.optim, optim_name)(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ],
            momentum=cfg.SOLVER.PROMPTSG.MOMENTUM
        )
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ]
        )
    else:
        optimizer = getattr(torch.optim, optim_name)(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ]
        )

    return optimizer
