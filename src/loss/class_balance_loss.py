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
        samples_per_cls : list[int]  # 각 클래스로부터의 원본 샘플 개수
        beta            : float      # 0.9~0.999 사이 권장 (0이면 일반 CE와 동일)
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
    def __init__(self, samples_per_cls, beta=0.9995, gamma=1.5, reduction="mean"):
        super().__init__()
        eff = 1.0 - np.power(beta, samples_per_cls)
        w = (1.0 - beta) / eff
        w = w / w.sum() * len(samples_per_cls)
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float32))
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.long().to(logits.device)
        weights = self.weights.to(device=logits.device, dtype=logits.dtype)

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = log_pt.exp().clamp_min(1e-8)                               # [B]

        cb_w = weights[targets]                                         # [B]
        loss = -cb_w * (1.0 - pt).pow(self.gamma) * log_pt              # [B]

        if self.reduction == "mean":
            return loss.sum() / (cb_w.sum() + 1e-12)  # ← CE와 동일 스케일
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss