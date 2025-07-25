import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassBalancedCrossEntropyLoss(nn.Module):
    """
    Class‑Balanced Cross‑Entropy (CB‑CE)
    """
    def __init__(self, samples_per_cls, beta=0.999, reduction="mean"):
        """
        samples_per_cls : list[int]
        beta            : float
        """
        super().__init__()
        eff_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(eff_num)
        weights = weights / weights.sum() * len(samples_per_cls)          # 정규화
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits  : [B, C]
        targets : [B]  (long)
        """
        loss = F.cross_entropy(
            logits, targets,
            weight=self.weights.to(logits.device),
            reduction=self.reduction
        )
        return loss
    
class ClassBalancedFocalLoss(nn.Module):
    """
    Class‑Balanced Focal Loss (CB‑Focal = CB‑weight ⨉ Focal(γ))
    """
    def __init__(self, samples_per_cls, beta=0.999, gamma=2.0, reduction="mean"):
        super().__init__()
        eff_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(eff_num)
        weights = weights / weights.sum() * len(samples_per_cls)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits  : [B, C]
        targets : [B]  (long)
        """
        weights = self.weights.to(logits.device)
        # one‑hot → 확률 p_t
        probs   = F.softmax(logits, dim=1)                 # [B, C]
        pt      = probs[torch.arange(len(targets)), targets]  # [B]

        focal_w = (1.0 - pt).pow(self.gamma)               # focal scaling
        cls_w   = weights[targets]                         # CB scaling
        loss    = -cls_w * focal_w * pt.log()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss