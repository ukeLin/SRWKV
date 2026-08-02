from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _resolve_split_dir(root: Path, split: str) -> Path:
    candidates = [root / split]
    if split == "val":
        candidates.extend([root / "valid", root / "validation", root / "test"])
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            return candidate
    expected = " or ".join(str(path / "{images,masks}") for path in candidates)
    raise FileNotFoundError(f"Could not find split folders. Expected {expected}")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        split_dir = _resolve_split_dir(self.data_path, split)
        image_files = _image_files(split_dir / "images")
        mask_files = _image_files(split_dir / "masks")
        mask_by_stem = {p.stem.replace("_segmentation", ""): p for p in mask_files}

        self.samples: list[tuple[Path, Path]] = []
        for image_path in image_files:
            key = image_path.stem.replace("_segmentation", "")
            mask_path = mask_by_stem.get(key)
            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No image-mask pairs found under {split_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.uniform(-30.0, 30.0)
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.float32) > 127).astype(np.int64)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            "image": image_tensor,
            "label": mask_tensor,
            "case_name": image_path.stem,
        }


ISICDataset = SegmentationDataset
