import argparse
import os
from argparse import BooleanOptionalAction

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from torchvision import transforms

from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset
from methods.brain_encoding.dino_vit import DINO_ViT_Encoder
from methods.brain_encoding.callbacks import WandbCorrCallback


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        output_size: int,
        learning_rate: float,
        clip_linear: bool = False,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.clip_linear = clip_linear
        if clip_linear:
            self.model = nn.Linear(1024, output_size)
        else:
            self.model = DINO_ViT_Encoder(output_size)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch, mode):
        if self.clip_linear:
            _, input, target = batch
        else:
            input, target, _ = batch
        pred = self(input)
        loss = F.mse_loss(pred, target)
        self.log_stat(f"{mode}_loss", loss, mode)
        return loss, pred

    def log_stat(self, name, stat, mode):
        self.log(
            name,
            stat,
            on_step=mode=='train',
            on_epoch=mode=='val',
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred = self.compute_loss(batch, "val")
        return pred
    
def load_data(cfg):

    if cfg.clip_linear:

        transform = transforms.ToTensor()
    
    else:

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    train_nsd = NaturalScenesDataset(
        root=cfg.dataset_dir,
        subject=cfg.subject,
        partition="train",
        transform=transform,
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
        return_average=cfg.predict_average,
    )

    val_nsd = NaturalScenesDataset(
        root=cfg.dataset_dir,
        subject=cfg.subject,
        partition="test",
        transform=transform,
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
        return_average=cfg.predict_average,
    )

    if cfg.clip_linear:

        train_set = NSDCLIPFeaturesDataset(
            nsd=train_nsd,
            clip_extractor_type='clip_2_0'
        )
        val_set = NSDCLIPFeaturesDataset(
            nsd=val_nsd,
            clip_extractor_type='clip_2_0'
        )

    else:

        train_set, val_set = train_nsd, val_nsd

    train_dataloader = data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = data.DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )
            
    return train_nsd, val_nsd, train_dataloader, val_dataloader

def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    roi_str = "_".join(cfg.roi) if isinstance(cfg.roi, list) else cfg.roi
    pred_str = "avg" if cfg.predict_average else "all"
    folder = "clip_linear" if cfg.clip_linear else "dino_vit"
    cfg.exp_name = f"{folder}/{cfg.subject:02d}_{roi_str}_{cfg.hemisphere[0]}_{pred_str}_{cfg.seed}"

    if cfg.roi == "hvc":
        cfg.roi = [
            "floc-faces", 
            "floc-words", 
            "floc-places", 
            "floc-bodies", 
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal"
        ]

    # Initialize dataset
    train_nsd, val_nsd, train_dataloader, val_dataloader = load_data(cfg)

    # Initialize brain encoder
    model = EncoderModule(
        output_size=1 if len(train_nsd.activations.shape) == 1 else train_nsd.activations.shape[1],
        learning_rate=cfg.learning_rate,
        clip_linear=cfg.clip_linear,
    )

    # Initialize callbacks
    os.makedirs(f"{cfg.ckpt_dir}/{cfg.exp_name}", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_dir}/{cfg.exp_name}",
        save_top_k=1,
        save_last=False,
        monitor="val_loss",
        mode="min",
    )
    
    callbacks = [checkpoint_callback]
    if not cfg.predict_average and cfg.roi == "all":
        r2_callback = WandbCorrCallback(locs=val_nsd.fs_coords[val_nsd.fs_indices][val_nsd.roi_indices], hemisphere=cfg.hemisphere, subjdir=val_nsd.subj_dir, clip_linear=cfg.clip_linear)
        callbacks.append(r2_callback)

    # Initialize loggers
    wandb.init(
        name=cfg.exp_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
    )
    wandb_logger = WandbLogger()
    logger = [wandb_logger]

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        deterministic=True,
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="hvc")
    parser.add_argument("--hemisphere", type=str, default="right")
    parser.add_argument("--predict_average", action=BooleanOptionalAction, default=False)
    parser.add_argument("--clip_linear", action=BooleanOptionalAction, default=False)

    # Training Parameters
    parser.add_argument("--dataset_dir", type=str, default="./data/NSD")
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # WandB Parameters
    parser.add_argument("--wandb_project", type=str, default="masters-thesis-encoder")
    parser.add_argument("--wandb_entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb_mode", type=str, default="online")

    # Parse arguments
    cfg = parser.parse_args()
    main(cfg)