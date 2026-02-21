# """
# Author: Yonglong Tian (yonglong@mit.edu)
# Date: May 07, 2020
# """
# from __future__ import print_function

# import torch
# import torch.nn as nn

# class SupConLoss(nn.Module):
#     def __init__(self, device):
#         super(SupConLoss, self).__init__()
#         self.device = device
#         self.temperature = 1.0
#     def forward(self, text_features, image_features, t_label, i_targets): 
#         batch_size = text_features.shape[0] 
#         batch_size_N = image_features.shape[0] 
#         mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
#             i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

#         logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach() 
#         exp_logits = torch.exp(logits) 
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
#         loss = - mean_log_prob_pos.mean()

#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, device, temperature=0.07, eps=1e-12, normalize=True):
#         super().__init__()
#         self.device = device
#         self.temperature = temperature
#         self.eps = eps
#         self.normalize = normalize

#     def forward(self, anchor_feat, cand_feat, anchor_label, cand_label):
#         if self.normalize:
#             anchor_feat = F.normalize(anchor_feat, dim=1)
#             cand_feat   = F.normalize(cand_feat,   dim=1)

#         B = anchor_feat.size(0)
#         N = cand_feat.size(0)

#         mask = torch.eq(
#             anchor_label.view(B, 1).expand(B, N),
#             cand_label.view(1, N).expand(B, N)
#         ).float().to(anchor_feat.device)

#         logits = (anchor_feat @ cand_feat.t()) / self.temperature
#         logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

#         log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True) + self.eps)

#         denom = mask.sum(dim=1) + self.eps
#         mean_log_prob_pos = (mask * log_prob).sum(dim=1) / denom

#         return -mean_log_prob_pos.mean()

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, eps=1e-12, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.normalize = normalize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, anchor_feat, cand_feat, anchor_label, cand_label):
        """
        Supervised Contrastive Loss (Equations 5-6 in PromptSG paper)
        
        Args:
            anchor_feat: (B1, D) - features của anchor
            cand_feat: (B2, D) - features của candidate  
            anchor_label: (B1,) - labels của anchor
            cand_label: (B2,) - labels của candidate
        """
        if self.normalize:
            anchor_feat = F.normalize(anchor_feat, dim=1)
            cand_feat = F.normalize(cand_feat, dim=1)

        B1, B2 = anchor_feat.size(0), cand_feat.size(0)
        
        # Similarity matrix
        sim_matrix = torch.matmul(anchor_feat, cand_feat.t()) / self.temperature  # (B1, B2)
        
        # Positive mask: same identity
        mask = torch.eq(
            anchor_label.unsqueeze(1).expand(B1, B2),
            cand_label.unsqueeze(0).expand(B1, B2)
        ).float().to(self.device)  # (B1, B2)
        
        # Log softmax over candidates
        logits = F.log_softmax(sim_matrix, dim=1)  # (B1, B2)
        
        # Loss: - sum over positive pairs, then average per anchor
        loss_pos = - (mask * logits).sum(dim=1)  # (B1,) sum over positives
        
        # Normalize by number of positives (avoid division by zero)
        num_pos = mask.sum(dim=1).clamp(min=1.0)
        loss_per_anchor = loss_pos / num_pos
        
        # Average over all anchors
        loss = loss_per_anchor.mean()
        
        return loss


def symmetric_supervised_contrastive_loss(v_feat, l_feat, labels, temperature=0.07):
    """
    Đúng Equation 4-5 trong paper
    """
    batch_size = v_feat.shape[0]
    
    # Normalize
    v_feat = F.normalize(v_feat, dim=1)
    l_feat = F.normalize(l_feat, dim=1)
    
    # Positive mask: P(i) trong paper
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    # Image-to-Text loss (Equation 5 đầu tiên)
    sim_i2t = torch.matmul(v_feat, l_feat.T) / temperature
    logits_i2t = F.log_softmax(sim_i2t, dim=1)
    
    # Sum over positive samples như paper
    loss_i2t = - (mask * logits_i2t).sum(dim=1)
    num_pos_per_row = mask.sum(dim=1)
    loss_i2t = loss_i2t / num_pos_per_row.clamp(min=1.0)
    loss_i2t = loss_i2t.mean()
    
    # Text-to-Image loss (Equation 5 thứ hai)
    sim_t2i = torch.matmul(l_feat, v_feat.T) / temperature
    logits_t2i = F.log_softmax(sim_t2i, dim=1)
    
    loss_t2i = - (mask * logits_t2i).sum(dim=1)
    loss_t2i = loss_t2i / num_pos_per_row.clamp(min=1.0)
    loss_t2i = loss_t2i.mean()
    
    # Symmetric loss (Equation 4)
    loss = (loss_i2t + loss_t2i)
    
    return loss



def symmetric_clip_contrastive_loss(image_feat: torch.Tensor, text_feat: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Standard CLIP-style symmetric InfoNCE for paired (image, text) samples.
    Positives are along the diagonal (i==j). No class labels needed.

    Args:
        image_feat: (B, D)
        text_feat:  (B, D)
        temperature: scalar
    """
    assert image_feat.ndim == 2 and text_feat.ndim == 2
    assert image_feat.shape[0] == text_feat.shape[0]
    B = image_feat.shape[0]
    image_feat = F.normalize(image_feat, dim=1)
    text_feat  = F.normalize(text_feat, dim=1)

    logits_i2t = (image_feat @ text_feat.t()) / temperature
    labels = torch.arange(B, device=logits_i2t.device, dtype=torch.long)
    loss_i2t = F.cross_entropy(logits_i2t, labels)

    logits_t2i = logits_i2t.t()
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    return 0.5 * (loss_i2t + loss_t2i)
