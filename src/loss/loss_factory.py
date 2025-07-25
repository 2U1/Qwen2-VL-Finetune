from .class_balance_loss import ClassBalancedCrossEntropyLoss, ClassBalancedFocalLoss
from .focal_loss import FocalLossCE
import torch.nn as nn

def get_loss_function(training_args, samples_per_class=None):
    
    if training_args.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif training_args.loss_type == "focal_loss":
        alpha = None if training_args.focal_alpha is None else [float(a) for a in training_args.focal_alpha.split(",")]
        return FocalLossCE(alpha=alpha, gamma=training_args.focal_gamma, reduction="mean")
    
    elif training_args.loss_type == "class_balanced_cross_entropy":
        return ClassBalancedCrossEntropyLoss(samples_per_cls=samples_per_class, beta=training_args.class_balanced_beta, reduction="mean")
    
    elif training_args.loss_type == "class_balanced_focal_loss":
        return ClassBalancedFocalLoss(samples_per_cls=samples_per_class, beta=training_args.class_balanced_beta, gamma=training_args.focal_gamma, reduction="mean")

    else:
        raise ValueError(f"Unknown loss type: {training_args.loss_type}")