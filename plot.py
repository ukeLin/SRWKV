from __future__ import annotations
import cv2
import numpy as np
import torch
from typing import Sequence

SYNAPSE_COLORMAP = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [255, 255, 0],
    7: [63, 208, 244],
    8: [241, 240, 234],
}

ACDC_COLORMAP = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
}

class2colormap = {
    9: SYNAPSE_COLORMAP,
    4: ACDC_COLORMAP
}

def make_rgb_darker(color: Sequence[int, int, int], percentage: float = 0.5) -> tuple[int, int, int]:
    def _dark(c: int) -> int:
        return int(max(0., c - c * percentage))
    r, g, b = color
    return _dark(r), _dark(g), _dark(b)

def is_grayscale(image: np.ndarray | torch.Tensor) -> bool:
    return not (len(image.shape) > 2 and image.shape[2] > 1)

def save_x_y(x: np.ndarray, y: np.ndarray, colormap: dict, out: str) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w];
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    for i, color in colormap.items():
        x = np.where(y == i, np.full_like(x, color), x)
    cv2.imwrite(out, x)

def save_x_y_hat(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, colormap: dict, out: str) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w]; y_hat: [h, w]
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8, y_hat.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_GRAY2BGR)
    for i, color in colormap.items():
        x = np.where(y_hat == i, np.full_like(x, color), x)
        contours, _ = cv2.findContours(np.array(y == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, contours, -1, make_rgb_darker(color, percentage=0.5), thickness=2)
    cv2.imwrite(out, x)

def save_x_y_tensor(x: torch.Tensor, y: torch.Tensor, colormap: dict, out: str) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w];
    """
    x = x if is_grayscale(x) else x.permute(1, 2, 0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    y = y.detach().cpu().numpy().astype(np.uint8)
    save_x_y(x, y, colormap, out)
