import argparse
import os
from argparse import BooleanOptionalAction
from typing import List, Union

import numpy as np
import open_clip
import pytorch_lightning as pl
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torchvision import transforms

from dataset.natural_scenes import NaturalScenesDataset
from dataset.nsd_induced import NSDInducedDataset
from model.feature_extractor import create_feature_extractor
from model.stable_diffusion import StableDiffusion
from prompt_optimization.optim_utils import (ConfigPromptOptimization,
                                             optimize_prompt)
from utils.configs import config_from_args


def slerp(embedding1, embedding2, val):
    embedding1 = embedding1[0]
    embedding2 = embedding2[0]
    low_norm = embedding1 / torch.norm(embedding1, dim=1, keepdim=True)
    high_norm = embedding2 / torch.norm(embedding2, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    faktor1 = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1).unsqueeze(0)
    mask = torch.isnan(faktor1)
    mean = torch.mean(faktor1[~mask])
    faktor1[mask] = mean
    faktor2 = (torch.sin(val * omega) / so).unsqueeze(1).unsqueeze(0)
    mask = torch.isnan(faktor2)
    mean = torch.mean(faktor2[~mask])
    faktor2[mask] = mean
    res = faktor1 * embedding1 + faktor2 * embedding2
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--hemisphere", type=str, default="right")

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

    # Captioning Parameters
    parser.add_argument("--neg-std-threshold", type=float, default=2)
    parser.add_argument("--pos-std-threshold", type=float, default=2)
    parser.add_argument("--num-captioned", type=int, default=50)
    parser.add_argument("--prompt-clip-model", type=str, default="ViT-L-14")
    parser.add_argument("--prompt-clip_pretrain", type=str, default="openai")
    parser.add_argument("--prompt-iterations", type=int, default=5000)
    parser.add_argument("--prompt-lr", type=float, default=1e-1)
    parser.add_argument("--prompt-weight-decay", type=float, default=1e-1)
    parser.add_argument("--prompt-prompt-len", type=int, default=16)
    parser.add_argument("--prompt-prompt-bs", type=int, default=1)
    parser.add_argument("--prompt-loss-weight", type=float, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--prompt-print-step", type=int, default=100)
    parser.add_argument(
        "--prompt-print-new-best", action=BooleanOptionalAction, default=False
    )

    # Synthesis Parameters
    parser.add_argument("--outputs-dir", type=str, default="./outputs/")
    parser.add_argument("--slerp-steps", type=int, default=25)
    parser.add_argument("--g", type=float, default=7.5)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # Parse arguments
    args = parser.parse_args()
    cfg = config_from_args(args, mode="pipeline")
    pl.seed_everything(cfg.seed, workers=True)

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

    # Select Max-EIs and Min-EIs
    print(f"Activation mean: {np.mean(y_pred)}, std: {np.std(y_pred)}")
    mean_eis = np.mean(y_pred)
    max_eis_idx = np.where(y_pred > mean_eis + cfg.pos_std_threshold * np.std(y_pred))[
        0
    ]
    min_eis_idx = np.where(y_pred < mean_eis - cfg.neg_std_threshold * np.std(y_pred))[
        0
    ]
    max_eis_dists = dataset.D[max_eis_idx, :][:, max_eis_idx]
    min_eis_dists = dataset.D[min_eis_idx, :][:, min_eis_idx]
    print(f"Max-EIs mean distance: {np.mean(max_eis_dists)}")
    print(f"Min-EIs mean distance: {np.mean(min_eis_dists)}")
    print(
        f"Max-EIs: {len(max_eis_idx)}, taking {min(len(max_eis_idx), cfg.num_captioned)}"
    )
    print(
        f"Min-EIs: {len(min_eis_idx)}, taking {min(len(min_eis_idx), cfg.num_captioned)}"
    )
    max_eis_idx = np.random.choice(
        max_eis_idx,
        min(len(max_eis_idx), cfg.num_captioned),
        replace=False,
    )
    min_eis_idx = np.random.choice(
        min_eis_idx,
        min(len(min_eis_idx), cfg.num_captioned),
        replace=False,
    )
    print(f"Max-EIs mean activation: {np.mean(y_pred[max_eis_idx])}")
    print(f"Min-EIs mean activation: {np.mean(y_pred[min_eis_idx])}")

    # Group captioning
    cfg_prompt = ConfigPromptOptimization(
        iter=cfg.prompt_iterations,
        lr=cfg.prompt_lr,
        weight_decay=cfg.prompt_weight_decay,
        prompt_len=cfg.prompt_prompt_len,
        prompt_bs=cfg.prompt_prompt_bs,
        loss_weight=cfg.prompt_loss_weight,
        batch_size=cfg.prompt_batch_size,
        print_step=cfg.prompt_print_step,
        print_new_best=cfg.prompt_print_new_best,
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.prompt_clip_model, pretrained=cfg.prompt_clip_pretrain, device=cfg.device
    )
    max_eis_target_images = [
        Image.open(os.path.join(nsd.root, nsd.df.iloc[idx]["filename"]))
        for idx in max_eis_idx
    ]
    min_eis_target_images = [
        Image.open(os.path.join(nsd.root, nsd.df.iloc[idx]["filename"]))
        for idx in min_eis_idx
    ]
    max_eis_learned_prompt = optimize_prompt(
        clip_model, clip_preprocess, max_eis_target_images, cfg_prompt, cfg.device
    )
    min_eis_learned_prompt = optimize_prompt(
        clip_model, clip_preprocess, min_eis_target_images, cfg_prompt, cfg.device
    )
    print(f"Max-EIs learned prompt: {max_eis_learned_prompt}")
    print(f"Min-EIs learned prompt: {min_eis_learned_prompt}")

    # Slerp interpolation and synthesis
    ldm = StableDiffusion(batch_size=1, device=cfg.device)
    condition1 = ldm.text_enc([max_eis_learned_prompt])
    condition2 = ldm.text_enc([min_eis_learned_prompt])
    steps = np.linspace(0, 1, cfg.slerp_steps)
    imgs = []
    activations = []
    for step in steps:
        condition = slerp(condition1, condition2, step)
        max_length = condition.shape[1]
        uncondition = ldm.text_enc([""], max_length)
        text_embedding = torch.cat([uncondition, condition])

        # Synthesize images
        with torch.no_grad():
            img = ldm.text_emb_to_img(
                text_embedding=text_embedding,
                return_pil=True,
                g=cfg.g,
                seed=cfg.seed,
                steps=cfg.inference_steps,
            )[0]
        imgs.append(img)

        # Predict activation
        img = transforms.ToTensor()(img)
        feats = feature_extractor(img.unsqueeze(0).to(cfg.device))
        activation = reg.predict(feats.cpu().detach().numpy())[0]
        activations.append(activation)

    # Save images
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outputs_dir, "max"), exist_ok=True)
    os.makedirs(os.path.join(cfg.outputs_dir, "min"), exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(os.path.join(cfg.outputs_dir, f"{i}_{np.round(activations[i].astype(np.float32), 4)}.png"))
    for i, img in enumerate(max_eis_target_images):
        img.save(os.path.join(cfg.outputs_dir, "max", f"{i}.png"))
    for i, img in enumerate(min_eis_target_images):
        img.save(os.path.join(cfg.outputs_dir, "min", f"{i}.png"))