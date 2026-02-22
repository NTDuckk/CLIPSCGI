import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss, symmetric_supervised_contrastive_loss, symmetric_clip_contrastive_loss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    
    # Initialize SupConLoss like stage1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supcon_loss = SupConLoss(device)
    
    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam=None, image_feat=None, text_feat=None):
            # ID loss
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                ID_LOSS = xent(score, target)
            else:
                ID_LOSS = F.cross_entropy(score, target)

            TRI_LOSS = torch.tensor(0.0, device=ID_LOSS.device)
            total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS

            method = getattr(cfg.MODEL, 'METHOD', 'promptsg')
            if method == 'clip_scgi':
                CON_LOSS = torch.tensor(0.0, device=ID_LOSS.device)
                if (image_feat is not None) and (text_feat is not None):
                    temp = getattr(cfg.MODEL.CLIPSCGI, 'TEMPERATURE', 0.07)
                    lam = getattr(cfg.MODEL.CLIPSCGI, 'LAMBDA_CON', 1.0)
                    CON_LOSS = symmetric_clip_contrastive_loss(image_feat, text_feat, temperature=temp)
                    total = total + lam * CON_LOSS
                return total, {
                    'id_loss': ID_LOSS.detach(),
                    'tri_loss': TRI_LOSS.detach(),
                    'con_loss': CON_LOSS.detach()
                }

            # PromptSG case (không dùng con_loss)
            SUPCON_LOSS = torch.tensor(0.0, device=ID_LOSS.device)
            return total, {
                'id_loss': ID_LOSS.detach(),
                'tri_loss': TRI_LOSS.detach(),
                'supcon_loss': SUPCON_LOSS.detach()
            }
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, image_feat=None, text_feat=None):
            # ID Loss (giống CLIP-ReID)
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = xent(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS) 
                else:   
                    TRI_LOSS = triplet(feat, target)[0]

                # Tính tổng loss cơ bản
                total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                method = getattr(cfg.MODEL, 'METHOD', 'promptsg')
                # =============
                # CLIP-SCGI: thêm CLIP-style paired contrastive loss (InfoNCE) giữa image_feat và caption/text_feat
                # =============
                if method == 'clip_scgi':
                    CON_LOSS = torch.tensor(0.0, device=ID_LOSS.device)
                    if (image_feat is not None) and (text_feat is not None):
                        temp = getattr(getattr(cfg.MODEL, 'CLIPSCGI', object()), 'TEMPERATURE', 0.07)
                        CON_LOSS = symmetric_clip_contrastive_loss(image_feat, text_feat, temperature=temp)
                        lam = getattr(getattr(cfg.MODEL, 'CLIPSCGI', object()), 'LAMBDA_CON', 1.0)
                        total_loss = total_loss + lam * CON_LOSS
                    return total_loss, {
                        'id_loss': ID_LOSS.detach(),
                        'tri_loss': TRI_LOSS.detach(),
                        'con_loss': CON_LOSS.detach(),
                    }

                # =============
                # PromptSG: thêm SupCon loss nếu có image_feat và text_feat
                # =============
                SUPCON_LOSS = torch.tensor(0.0, device=ID_LOSS.device)

                prompt_mode = getattr(cfg.MODEL.PROMPTSG, "PROMPT_MODE", "composed")
                enable_supcon_in_simplified = getattr(cfg.MODEL.PROMPTSG, "ENABLE_SUPCON_IN_SIMPLIFIED", False)

                if (image_feat is not None) and (text_feat is not None):
                    if (prompt_mode != "simplified") or enable_supcon_in_simplified:
                        # Use symmetric supervised contrastive loss (Equation 4-5)
                        SUPCON_LOSS = symmetric_supervised_contrastive_loss(
                            image_feat, text_feat, target, temperature=0.07
                        )
                        total_loss = total_loss + cfg.MODEL.PROMPTSG.LAMBDA_SUPCON * SUPCON_LOSS
                
                return total_loss, {
                    'id_loss': ID_LOSS.detach(),
                    'tri_loss': TRI_LOSS.detach(),
                    'supcon_loss': SUPCON_LOSS.detach()
                }

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    
    return loss_func