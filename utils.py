from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ("module.", "_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value

    msg = model.load_state_dict(cleaned, strict=strict)
    return {
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def dice_miou(logits: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    gt = label[:, 0].long()
    pred_fg = pred == 1
    gt_fg = gt == 1
    intersection = torch.logical_and(pred_fg, gt_fg).sum().float()
    union = torch.logical_or(pred_fg, gt_fg).sum().float()
    dice = (2.0 * intersection + 1e-5) / (pred_fg.sum().float() + gt_fg.sum().float() + 1e-5)
    miou = (intersection + 1e-5) / (union + 1e-5)
    return dice.item(), miou.item()


def soft_binary_dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = prob.float()
    target = target.float()
    dims = (1, 2, 3)
    intersection = torch.sum(prob * target, dim=dims)
    denominator = torch.sum(prob * prob, dim=dims) + torch.sum(target * target, dim=dims)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 1.0 - dice.mean()


def shape_prior_loss(shape_prior: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    target = (label > 0).float()
    if shape_prior.dim() == 3:
        shape_prior = shape_prior.unsqueeze(1)
    if shape_prior.shape[1] > 1:
        shape_prior = shape_prior[:, 1:2]
    if shape_prior.shape[2:] != target.shape[2:]:
        shape_prior = F.interpolate(shape_prior, size=target.shape[2:], mode="bilinear", align_corners=False)
    shape_prior = torch.clamp(shape_prior.float(), min=1e-6, max=1.0 - 1e-6)
    bce = F.binary_cross_entropy(shape_prior, target)
    dice = soft_binary_dice_loss(shape_prior, target)
    return 0.5 * bce + 0.5 * dice
