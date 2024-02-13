import argparse
import os
from argparse import BooleanOptionalAction

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils import data

from dataset.natural_scenes import NaturalScenesDataset
from dataset.nsd_induced import NSDInducedDataset
from model.brain_encoder import EncoderModule
from model.feature_extractor import create_feature_extractor
from utils.callbacks import WandbR2Callback, WandbTSNECallback
from utils.configs import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="floc-faces")
    parser.add_argument("--hemisphere", type=str, default="right")
    parser.add_argument("--feature-extractor-type", type=str, default="clip")
    parser.add_argument("--n-neighbors", type=int, default=0)
    parser.add_argument("--distance-metric", type=str, default="cosine")
    parser.add_argument("--predict-average", action=BooleanOptionalAction, default=True)

    # Training Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument("--logs-dir", type=str, default="./logs/")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--max-epochs", type=int, default=50)
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

    # Parse arguments
    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.exp_name is None:
        roi_str = "_".join(cfg.roi) if isinstance(cfg.roi, list) else cfg.roi
        pred_str = "avg" if cfg.predict_average else "all"
        cfg.exp_name = f"{cfg.subject:02d}_{roi_str}_{cfg.hemisphere[0]}_{cfg.feature_extractor_type}_{cfg.n_neighbors}_{cfg.distance_metric}_{pred_str}_{cfg.seed}"

    # Initialize feature extractor
    feature_extractor = create_feature_extractor(cfg.feature_extractor_type, cfg.device)

    # Initialize dataset
    nsd = NaturalScenesDataset(
        root=cfg.data_dir,
        subject=cfg.subject,
        partition="train",
        roi=cfg.roi,
        hemisphere=cfg.hemisphere,
        return_coco_id=False
    )
    dataset = NSDInducedDataset(
        nsd=nsd,
        feature_extractor=feature_extractor,
        predict_average=cfg.predict_average,
        metric=cfg.distance_metric,
        n_neighbors=cfg.n_neighbors,
        seed=cfg.seed,
        batch_size_feature_extraction=cfg.batch_size,
    )

    # Initialize dataloaders (split into train and validation sets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    val_set.dataset.split = "test"
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

    # Initialize brain encoder
    model = EncoderModule(
        feature_extractor=feature_extractor,
        input_size=feature_extractor.feature_size,
        output_size=dataset.target_size,
        learning_rate=cfg.learning_rate,
        brain_wise_mean=dataset.targets_mean,
        brain_wise_std=dataset.targets_std,
    )

    # Initialize callbacks
    os.makedirs(f"{cfg.ckpt_dir}/{cfg.exp_name}", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_dir}/{cfg.exp_name}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    tsne_callback = WandbTSNECallback(cfg.predict_average)
    callbacks = [checkpoint_callback, tsne_callback]
    if not cfg.predict_average:
        callbacks.append(WandbR2Callback())

    # Initialize loggers
    csv_logger = CSVLogger(cfg.logs_dir, name=cfg.exp_name, flush_logs_every_n_steps=1)
    wandb.init(
        name=cfg.exp_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
    )
    wandb_logger = WandbLogger()
    logger = [wandb_logger, csv_logger]

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        deterministic=True,
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
