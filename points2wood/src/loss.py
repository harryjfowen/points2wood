import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 gamma: float = 2.0,
                 alpha: float = None,
                 reduction: str = "none",
                 weight: Tensor = None,
                 label_smoothing: float = None):  # Add label_smoothing parameter
        
        super(Poly1FocalLoss, self).__init__()
        self.epsilon  = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing  # Store label_smoothing

    def forward(self, logits, labels, labelweights = None):

        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = torch.sigmoid(logits)

        bce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                      target=labels,
                                                      reduction="none",
                                                      weight=self.weight,
                                                      pos_weight=None)

        pt = labels * p + (1 - labels) * (1 - p)

        FL = bce_loss * ((1 - pt + 1e-8) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1, self.gamma
    
