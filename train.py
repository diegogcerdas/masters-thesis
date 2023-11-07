import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils import data

from dataset import NaturalScenesDataset
from model import BrainDiVEModule
from utils import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="OFA")
    parser.add_argument("--hemisphere", type=str, default="left")

    # Training Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument("--logs-dir", type=str, default="./logs/")
    parser.add_argument("--exp-name", type=str, default="default-run")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )
    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")

    pl.seed_everything(cfg.seed)

    dataset = NaturalScenesDataset(
        root=cfg.data_dir,
        subject=cfg.subject,
        partition="train",
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    if cfg.resume_ckpt is not None:
        print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        model = BrainDiVEModule.load_from_checkpoint(cfg.resume_ckpt)
        assert model.subject == cfg.subject
        assert model.roi == cfg.roi
    else:
        model = BrainDiVEModule(
            subject=cfg.subject,
            roi=cfg.roi,
            hemisphere=cfg.hemisphere,
            num_voxels=dataset.num_voxels,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_dir}/{cfg.exp_name}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    os.makedirs(f"{cfg.ckpt_dir}/{cfg.exp_name}", exist_ok=True)
    callbacks = [checkpoint_callback]

    csv_logger = CSVLogger(cfg.logs_dir, name=cfg.exp_name, flush_logs_every_n_steps=1)
    logger = [csv_logger]

    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=2,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.resume_ckpt)
