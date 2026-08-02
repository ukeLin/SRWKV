from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from model import build_model
from utils import dice_miou, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SRWKV for binary medical image segmentation.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(args.data_path, split="val", img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(3, 2, img_size=args.img_size).to(device)
    msg = load_checkpoint(model, args.checkpoint)
    print(f"missing keys: {len(msg['missing_keys'])}, unexpected keys: {len(msg['unexpected_keys'])}")
    model.eval()

    rows = []
    total_dice = 0.0
    total_miou = 0.0
    total_images = 0

    for batch in tqdm(loader, desc="test"):
        image = batch["image"].to(device, non_blocking=True).float()
        label = batch["label"].to(device, non_blocking=True).long()
        output = model(image)
        logits = output[0] if isinstance(output, tuple) else output
        dice, miou = dice_miou(logits, label)
        batch_size = image.shape[0]
        total_dice += dice * batch_size
        total_miou += miou * batch_size
        total_images += batch_size
        for name in batch["case_name"]:
            rows.append({"case_name": name, "dice": dice, "miou": miou})

    mean_dice = total_dice / max(total_images, 1)
    mean_miou = total_miou / max(total_images, 1)
    print(f"Dice: {mean_dice:.4f} ({mean_dice * 100:.2f}%)")
    print(f"mIoU: {mean_miou:.4f} ({mean_miou * 100:.2f}%)")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case_name", "dice", "miou"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
