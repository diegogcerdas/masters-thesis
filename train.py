import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils import data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from dataset import NaturalScenesDataset
from model import BrainEncoderModule

from utils import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Data Parameters

    # TODO: Model Parameters

    # Training Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument("--logs-dir", type=str, default="./logs/")
    parser.add_argument("--exp-name", type=str, default="default-run")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
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
    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")

    pl.seed_everything(cfg.seed)

    train_set = NaturalScenesDataset()
    train_loader = data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_set = NaturalScenesDataset()
    val_loader = data.DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    if cfg.resume_ckpt is not None:
        print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        model = BrainEncoderModule.load_from_checkpoint(cfg.resume_ckpt)
    else:
        model = BrainEncoderModule(
            learning_rate=cfg.learning_rate, 
            weight_decay=cfg.weight_decay
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_dir}/{cfg.exp_name}",
        save_top_k=1,
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
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.resume_ckpt)