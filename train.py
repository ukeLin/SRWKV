from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from loss import DiceCELoss
from model import build_model
from utils import dice_miou, set_seed, shape_prior_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRWKV for binary medical image segmentation.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--work-dir", default="runs/srwkv")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sp-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def unpack_output(output):
    if isinstance(output, tuple):
        return output
    return output, None


def compute_loss(logits, shape_prior, label, criterion, sp_weight):
    pred_loss = criterion(logits, label)
    if shape_prior is None or sp_weight <= 0:
        sp_loss = pred_loss.new_tensor(0.0)
        return pred_loss, pred_loss, sp_loss
    sp_loss = shape_prior_loss(shape_prior, label)
    return pred_loss + sp_weight * sp_loss, pred_loss, sp_loss


def run_epoch(model, loader, criterion, optimizer, device, sp_weight, amp):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and is_train)
    totals = {"loss": 0.0, "pred_loss": 0.0, "sp_loss": 0.0, "dice": 0.0, "miou": 0.0}
    count = 0
    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)

    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True).float()
        label = batch["label"].to(device, non_blocking=True).long()
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits, shape_prior = unpack_output(model(image))
                loss, pred_loss, sp_loss = compute_loss(logits, shape_prior, label, criterion, sp_weight)
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

        dice, miou = dice_miou(logits.detach(), label)
        batch_size = image.shape[0]
        totals["loss"] += loss.item() * batch_size
        totals["pred_loss"] += pred_loss.item() * batch_size
        totals["sp_loss"] += sp_loss.item() * batch_size
        totals["dice"] += dice * batch_size
        totals["miou"] += miou * batch_size
        count += batch_size
        pbar.set_postfix(loss=totals["loss"] / count, dice=totals["dice"] / count)

    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    work_dir = Path(args.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set = SegmentationDataset(args.data_path, split="train", img_size=args.img_size, augment=True)
    val_set = SegmentationDataset(args.data_path, split="val", img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pretrained = args.pretrained if args.pretrained and Path(args.pretrained).is_file() else None
    if args.pretrained and pretrained is None:
        print(f"Pretrained weight not found: {args.pretrained}. Training from scratch.")

    model = build_model(3, 2, img_size=args.img_size, encoder_pretrained_path=pretrained).to(device)
    criterion = DiceCELoss(ce_weight=0.5, dc_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"Resumed from {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.sp_weight, args.amp)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, args.sp_weight, False)
        scheduler.step()
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} miou {train_metrics['miou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f} miou {val_metrics['miou']:.4f}"
        )

        state = {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        }
        torch.save(state, checkpoint_dir / "latest.pth")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            state["best_dice"] = best_dice
            torch.save(state, checkpoint_dir / "best.pth")
            print(f"saved best checkpoint: dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
