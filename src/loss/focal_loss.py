import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLossCE(nn.Module):
    """
    Focal Loss for multi‑class
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits : [B, C]
        targets: [B]  (long)
        """
        weight = None
        if self.alpha is not None and self.alpha.numel() > 0:
            weight = self.alpha.to(logits.device)
        
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction="none", 
            weight=weight
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal
