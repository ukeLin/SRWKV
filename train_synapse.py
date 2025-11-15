from __future__ import annotations
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os.path import join
from collections import defaultdict
import torch
import numpy as np
import monai
from monai import data
from monai.metrics import CumulativeAverage
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger
from lr_scheduler import LR_SCHEDULERS
from loss import LOSSES
from eval import eval_single_volume
from model import build_model
from dataset_synapse import SynapseDataset
from torchvision.transforms import transforms
from typing import Callable

torch.set_float32_matmul_precision("medium")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW
}

class Synapse(L.LightningModule):
    def __init__(self, name: str) -> None:
        super(Synapse, self).__init__()
        self.name = name
        self.num_classes = 9
        self.max_epochs = 300
        self.freeze_encoder_epochs = 0

        self._model = build_model(
            in_channels=3,  
            num_classes=self.num_classes,
            encoder_pretrained_path="/root/autodl-tmp/SRWKV/model/pretrained/net_S.pth"
        ).to(device)

        self.build_loss()
        self.tl_metric = CumulativeAverage()
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def forward(self, x: torch.Tensor):
        return self._model(x)

    def prepare_data(self) -> None:
        self.norm_x_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = SynapseDataset(
            base_dir="/root/autodl-tmp/dataset/Synapse/train_npz",
            split="train",
            norm_x_transform=self.norm_x_transform,
            norm_y_transform=transforms.ToTensor(),
        )

        self.val_dataset = SynapseDataset(base_dir="/root/autodl-tmp/dataset/Synapse/test_vol_h5", split="test_vol")

    def train_dataloader(self) -> data.DataLoader:
        tdl_0 = {
            "batch_size": 24,
            "num_workers": 6,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "worker_init_fn": None
        }

        return data.DataLoader(self.train_dataset, **tdl_0)

    def val_dataloader(self) -> data.DataLoader:
        vdl_0 = {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 4,
            "persistent_workers": True
        }

        return data.DataLoader(self.val_dataset, **vdl_0)

    def build_loss(self):
        loss_0 = ("DiceCELoss", {
            "ce_weight": 0.5,
            "dc_weight": 0.5,
        })

        self.loss = LOSSES[loss_0[0]](**loss_0[1])
        self.aux_loss = torch.nn.BCELoss()
        self.aux_weight = 0.1

    @property
    def criterion(self) -> Callable[..., torch.Tensor]:
        return self.loss

    def configure_optimizers(self) -> dict:
        optimizer_0 = ("AdamW", {
            "lr": 5e-4,
            "weight_decay": 1e-3,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999)
        })
        optimizer = OPTIMIZERS[optimizer_0[0]](self._model.parameters(), **optimizer_0[1])

        scheduler_1 = ("CosineAnnealingLR", {
            "T_max": self.max_epochs,
            "eta_min": 1e-6
        })
        scheduler = LR_SCHEDULERS[scheduler_1[0]](optimizer, **scheduler_1[1])

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def log_and_logger(self, name: str, value: ...) -> None:
        self.log(name, value)
        logger.info(f"epoch: {self.current_epoch} - {name}: {value}")

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.freeze_encoder_epochs:
            self._model.freeze_encoder()
            logger.info(f"Epoch {self.current_epoch}: Encoder is FROZEN")
        else:
            self._model.unfreeze_encoder()
            if self.current_epoch == self.freeze_encoder_epochs:
                logger.info(f"Epoch {self.current_epoch}: Encoder is UNFROZEN")
        super().on_train_epoch_start()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, label = batch["image"].to(device), batch["label"]
        label = label.to(device)

        pred, saliency_mask1 = self.forward(image)
        loss = self.criterion(pred, label)

        if label.dim() == 3:
            label = label.unsqueeze(1)
        saliency_gt = (label > 0).float()
        
        if saliency_gt.shape[1] != 1:
            saliency_gt = saliency_gt.sum(dim=1, keepdim=True).clamp(0, 1)
        
        if saliency_mask1.shape[2:] != saliency_gt.shape[2:]:
            saliency_gt = torch.nn.functional.interpolate(
                saliency_gt, size=saliency_mask1.shape[2:], mode='bilinear', align_corners=False
            )
        
        aux_loss = self.aux_loss(saliency_mask1, saliency_gt)
        total_loss = loss + self.aux_weight * aux_loss

        self.log("loss", total_loss.item(), prog_bar=True)
        self.log("main_loss", loss.item())
        self.log("aux_loss", aux_loss.item())
        
        self.tl_metric.append(total_loss.item())
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return total_loss

    def on_train_epoch_end(self) -> None:
        tl = self.tl_metric.aggregate()
        if hasattr(tl, 'item'):
            tl = tl.item()
        
        self.log_and_logger("mean_train_loss", tl)
        self.tl_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        volume, label = batch["image"], batch["label"]
        metric = eval_single_volume(
            model=self._model,
            volume=volume,
            label=label,
            num_classes=self.num_classes,
            output=join(self.name, str(self.current_epoch)),
            patch_size=(224, 224),
            device=device,
            norm_x_transform=getattr(self, "norm_x_transform", None),
        )

        for metric_name, class_metric in metric.items():
            for class_name, value in class_metric.items():
                self.vs_metric[metric_name][class_name].append(np.mean(value))

    def on_validation_epoch_end(self) -> None:
        for metric_name, class_metric in self.vs_metric.items():
            avg_metric = []
            for class_name, value in class_metric.items():
                t = np.mean(value)
                self.log(f"val_{metric_name}_{class_name}", t)
                avg_metric.append(t)
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(avg_metric))
        self.vs_metric = defaultdict(lambda: defaultdict(list))

def train(name: str) -> None:
    os.makedirs(name, exist_ok=True)
    logger.add(join(name, "training.log"))

    model = Synapse(name)


    checkpoint_callback = ModelCheckpoint(
        dirpath=join(name, "checkpoints"),
        monitor="val_mean_dice",
        mode="max",
        filename="{epoch:02d}-{val_mean_dice:.4f}",
        save_last=True,
        save_top_k=1
    )
    
    checkpoint_callback_high_dice = ModelCheckpoint(
        dirpath=join(name, "checkpoints", "high_dice"),
        monitor="val_mean_dice",
        mode="max",
        filename="{epoch:02d}-{val_mean_dice:.4f}",
        save_top_k=-1,  
        save_last=False,
        every_n_epochs=20  
    )

    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        mode="max",
        min_delta=0.00,
        patience=15
    )

    trainer = L.Trainer(
        precision=32,
        accelerator=device,
        devices="auto",
        max_epochs=model.max_epochs,
        check_val_every_n_epoch=20,
        gradient_clip_val=None,
        default_root_dir=name,
        callbacks=[checkpoint_callback, checkpoint_callback_high_dice, early_stop_callback],
        enable_checkpointing=True
    )

    # 从最新的checkpoint恢复训练
    # last_ckpt = join(name, "checkpoints", "epoch=219-val_mean_dice=0.8266.ckpt")
    # if os.path.exists(last_ckpt):
    #     print(f"✓ 从checkpoint恢复训练: {last_ckpt}")
    #     trainer.fit(model, ckpt_path=last_ckpt)
    # else:
    #     print("✓ 未找到checkpoint，从头开始训练")
    #     trainer.fit(model, ckpt_path=None)

    trainer.fit(model, ckpt_path=None)

if __name__ == "__main__":
    L.seed_everything(42)
    monai.utils.set_determinism(42)
    train("log/SRWKV-synapse")
