import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossCE(nn.Module):
    """
    Plain Focal Loss (multi-class), optional class weights alpha
    - gamma=0, alpha=None  → nn.CrossEntropyLoss(mean)와 동일 스케일
    - gamma=0, alpha!=None → nn.CrossEntropyLoss(weight=alpha, mean)와 동일 스케일
    """
    def __init__(self, alpha=None, gamma=1.5, reduction="mean"):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        # 빈 텐서는 "가중치 없음" 신호로 사용
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  [B, C]
        targets: [B] (long)
        """
        targets = targets.long().to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)                           # [B, C]
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B]
        pt = log_pt.exp().clamp_min(1e-8)                                   # [B]

        if self.alpha.numel() > 0:
            alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[targets]  # [B]
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt               # [B]

        if self.reduction == "mean":
            # CE의 "가중 평균"과 스케일 일치: 분모를 가중치 합으로
            denom = (alpha_t.sum() if self.alpha.numel() > 0 else pt.new_tensor(len(pt)))
            return loss.sum() / (denom + 1e-12)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
