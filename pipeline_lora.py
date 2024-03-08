import argparse
import torch
from utils.configs import config_from_args
from argparse import BooleanOptionalAction

import argparse
import os
from argparse import BooleanOptionalAction

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from dataset.natural_scenes import NaturalScenesDataset
from dataset.nsd_induced import NSDInducedDataset
from model.feature_extractor import create_feature_extractor
from utils.configs import config_from_args

from torchvision import transforms
import pandas as pd

from model.lora_dreambooth import LoRADreamBooth
from dataset.dreambooth import DreamBoothDataset
import torch.utils.data as data
from utils.img_utils import save_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--hemisphere", type=str, default="right")
    parser.add_argument("--outputs-dir", type=str, default="./outputs/")

    # Three different ways to specify the ROI:
    # 1. A predefined ROI
    parser.add_argument("--roi", type=str, default=None)
    # 2. A voxel neighborhood
    parser.add_argument("--center-voxel", type=int, default=None)
    parser.add_argument("--n-neighbor-voxels", type=int, default=None)
    # 3. A list of voxels
    parser.add_argument("--voxels-filename", type=str, default=None)

    # Model Parameters
    parser.add_argument("--feature-extractor-type", type=str, default="clip")
    parser.add_argument("--distance-metric", type=str, default="cosine")
    parser.add_argument("--n-neighbors", type=int, default=0)

    # MEI Parameters
    parser.add_argument("--pos-std-threshold", type=float, default=2)
    parser.add_argument("--num-captioned", type=int, default=100)

    # LoRA Parameters
    parser.add_argument("--pretrained_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_prompt", type=str, default="<dgcmt>")
    parser.add_argument("--validation_prompt", type=str, default="<dgcmt>")
    parser.add_argument("--max_train_epochs", type=int, default=50)
    parser.add_argument("--num_validation_images", type=int, default=20)
    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--train_text_encoder", action=BooleanOptionalAction, default=True)
    parser.add_argument("--inference_steps", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="lora")
    pl.seed_everything(cfg.seed, workers=True)

    # Setup output directory
    max_dir = os.path.join(cfg.outputs_dir, "max")
    lora_dir = os.path.join(cfg.outputs_dir, "lora")
    os.makedirs(max_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # Initialize feature extractor
    feature_extractor = create_feature_extractor(cfg.feature_extractor_type, cfg.device)

    # Initialize dataset
    voxel_idx = (
        np.load(cfg.voxels_filename).astype(np.int64).tolist()
        if cfg.voxels_filename is not None
        else None
    )
    nsd = NaturalScenesDataset(
        root=cfg.data_dir,
        subject=cfg.subject,
        partition="train",
        hemisphere=cfg.hemisphere,
        roi=cfg.roi,
        center_voxel=cfg.center_voxel,
        n_neighbor_voxels=cfg.n_neighbor_voxels,
        voxel_idx=voxel_idx,
        return_coco_id=False,
    )
    dataset = NSDInducedDataset(
        nsd=nsd,
        feature_extractor=feature_extractor,
        predict_average=True,
        metric=cfg.distance_metric,
        n_neighbors=cfg.n_neighbors,
        seed=cfg.seed,
        keep_features=True,
    )

    # RSA before Linear Regression
    voxel_RDM = (dataset.targets.reshape(-1, 1) - dataset.targets.reshape(1, -1)).abs()
    idx = torch.triu_indices(*voxel_RDM.shape, offset=1)
    voxel_RDM = voxel_RDM[idx[0], idx[1]].numpy()
    rep_RDM = dataset.D[idx[0], idx[1]]
    r = spearmanr(voxel_RDM, rep_RDM).statistic
    print(f"RSA before Linear Regression: {round(r, 4)}")

    # Linear regression
    reg = LinearRegression().fit(dataset.features, dataset.targets)
    y_pred = reg.predict(dataset.features)
    metric = reg.score(dataset.features, dataset.targets)
    print(f"R^2: {round(metric, 4)}")

    # RSA after Linear Regression
    voxel_RDM = np.abs(y_pred.reshape(-1, 1) - y_pred.reshape(1, -1))
    voxel_RDM = voxel_RDM[idx[0], idx[1]]
    r = spearmanr(voxel_RDM, rep_RDM).statistic
    print(f"RSA after Linear Regression: {round(r, 4)}")

    # Select Max-EIs
    max_eis_idx = np.where(y_pred > np.mean(y_pred) + cfg.pos_std_threshold * np.std(y_pred))[0]
    max_eis_dists = dataset.D[max_eis_idx, :][:, max_eis_idx]
    max_eis_idx_sample = np.random.choice(
        max_eis_idx,
        min(len(max_eis_idx), cfg.num_captioned),
        replace=False,
    )
    max_eis_target_images = [
        Image.open(os.path.join(nsd.root, nsd.df.iloc[idx]["filename"]))
        for idx in max_eis_idx_sample
    ]
    print(f"Max-EIs activation mean: {np.mean(y_pred)}, std: {np.std(y_pred)}.")
    print(f"Max-EIs mean distance: {np.mean(max_eis_dists)}.")
    print(f"Number of Max-EIs: {len(max_eis_idx)}, sampled {len(max_eis_idx_sample)}.")
    print(f"Sampled Max-EIs mean activation: {np.mean(y_pred[max_eis_idx_sample])}, std {np.std(y_pred[max_eis_idx_sample])}.")
    
    # Save Max-EIs
    save_images(max_eis_target_images, max_dir)

    # Perform LoRA
    train_dataset = DreamBoothDataset(
        instance_data_root=max_dir,
        size=cfg.resolution,
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    lora = LoRADreamBooth(
        pretrained_model_name_or_path=cfg.pretrained_path,
        rank=cfg.rank,
        train_text_encoder=cfg.train_text_encoder,
        instance_prompt=cfg.instance_prompt,
        validation_prompt=cfg.validation_prompt,
        num_validation_images=cfg.num_validation_images,
        validation_epochs=cfg.validation_epochs,
        max_train_epochs=cfg.max_train_epochs,
        inference_steps=cfg.inference_steps,
        output_dir=lora_dir,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        deterministic=True,
        devices=1,
        max_epochs=cfg.max_train_epochs,
        gradient_clip_val=cfg.max_grad_norm,
        gradient_clip_algorithm="norm",
    )
    # Train model
    trainer.fit(lora, train_dataloader)

    # Predict activations
    data = []
    for folder in os.listdir(lora_dir):
        # Skip non-folders
        if not os.path.isdir(os.path.join(lora_dir, folder)):
            continue
        acts = []
        for i in range(cfg.num_validation_images):
            img = Image.open(os.path.join(lora_dir, folder, f'{i}.png'))
            img = transforms.ToTensor()(img)
            if img.sum() == 0:
                acts.append(np.nan)
                continue
            feats = feature_extractor(img.unsqueeze(0).to(cfg.device))
            activation = reg.predict(feats.cpu().detach().numpy())[0]
            acts.append(activation)
        data.append((folder, acts))
        print(f"{folder}: mean {np.nanmean(acts)}, std {np.nanstd(acts)}")
    df = pd.DataFrame(data, columns=["folder", "activations"])
    df.to_csv(os.path.join(lora_dir, "activations.csv"), index=False)

