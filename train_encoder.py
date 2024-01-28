import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils import data

from dataset.natural_scenes import NaturalScenesDataset
from model.encoder.encoder_module import EncoderModule
from utils.configs import config_from_args

import wandb
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default=["OFA"])
    parser.add_argument("--hemisphere", type=str, default="left")
    parser.add_argument("--feature-extractor-type", type=str, default="clip")
    parser.add_argument("--encoder-type", type=str, default="linear")

    # Training Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument("--logs-dir", type=str, default="./logs/")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-start", type=float, default=1e-1)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # WandB Parameters
    parser.add_argument("--wandb-project", type=str, default="masters-thesis")
    parser.add_argument("--wandb-entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb-mode", type=str, default="online")

    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.exp_name is None:
        roi_str = "_".join(cfg.roi) if isinstance(cfg.roi, list) else cfg.roi
        cfg.exp_name = f"{cfg.subject:02d}_{roi_str}_{cfg.hemisphere[0]}_{cfg.feature_extractor_type}_{cfg.encoder_type}_{cfg.seed}"

    dataset = NaturalScenesDataset(
        root=cfg.data_dir,
        subject=cfg.subject,
        partition="train",
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        shuffle=False,
    )

    model = EncoderModule(
        subject=cfg.subject,
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
        num_voxels=dataset.num_voxels,
        feature_extractor_type=cfg.feature_extractor_type,
        encoder_type=cfg.encoder_type,
        learning_rate=cfg.lr_start,
        lr_gamma=(cfg.lr_end / cfg.lr_start) ** (1 / (cfg.max_epochs - 1)),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_dir}/{cfg.exp_name}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    os.makedirs(f"{cfg.ckpt_dir}/{cfg.exp_name}", exist_ok=True)

    csv_logger = CSVLogger(cfg.logs_dir, name=cfg.exp_name, flush_logs_every_n_steps=1)

    wandb.init(
        name=cfg.exp_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
    )
    wandb_logger = WandbLogger()

    logger = [wandb_logger, csv_logger]
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        deterministic=True,
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)
