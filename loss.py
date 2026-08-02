from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def one_hot(target: Tensor, num_classes: int) -> Tensor:
    return torch.cat([(target == i).unsqueeze(1) for i in range(num_classes)], dim=1).float()


def binary_dice_loss(prob: Tensor, target: Tensor) -> Tensor:
    target = target.float()
    intersection = torch.sum(prob * target)
    denominator = torch.sum(prob * prob) + torch.sum(target * target)
    return 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def multiclass_dice_loss(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    softmax: bool = True,
) -> Tensor:
    num_classes = logits.shape[1]
    prob = torch.softmax(logits, dim=1) if softmax else logits
    target_oh = one_hot(target, num_classes)
    if weight is None:
        weight = torch.ones(num_classes, device=logits.device, dtype=logits.dtype)
    loss = sum(binary_dice_loss(prob[:, i], target_oh[:, i]) * weight[i] for i in range(num_classes))
    return loss / num_classes


class DiceCELoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dc_weight: float = 1.0,
        softmax: bool = True,
        ce_class_weights: Optional[Tensor] = None,
        dc_class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dc_weight = dc_weight
        self.softmax = softmax
        self.dc_class_weights = dc_class_weights
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights) if softmax else nn.NLLLoss(weight=ce_class_weights)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target[:, 0].long()
        ce = self.ce(logits, target) * self.ce_weight
        dice = multiclass_dice_loss(logits, target, self.dc_class_weights, self.softmax) * self.dc_weight
        return ce + dice
