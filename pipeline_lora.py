import argparse
import os
from argparse import BooleanOptionalAction

import numpy as np
import pytorch_lightning as pl
import torch

from models.feature_extractor import create_feature_extractor
from pipeline.brain_encoder import encode, predict_from_dir
from pipeline.data import load_dataset
from pipeline.lora import perform_lora
from pipeline.stimuli import save_stimuli
from utils.configs import config_from_args

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
    parser.add_argument("--num-stimuli", type=int, default=100)

    # LoRA Parameters
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--instance-prompt", type=str, default="photo of <dgcmt>")
    parser.add_argument("--validation-prompt", type=str, default="photo of <dgcmt>")
    parser.add_argument("--max-train-epochs", type=int, default=50)
    parser.add_argument("--num-validation-images", type=int, default=25)
    parser.add_argument("--validation-epochs", type=int, default=10)
    parser.add_argument(
        "--train-text-encoder", action=BooleanOptionalAction, default=True
    )
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=18)
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

    dataset = load_dataset(
        data_dir=cfg.data_dir,
        subject=cfg.subject,
        hemisphere=cfg.hemisphere,
        roi=cfg.roi,
        center_voxel=cfg.center_voxel,
        n_neighbor_voxels=cfg.n_neighbor_voxels,
        voxels_filename=cfg.voxels_filename,
        feature_extractor_type=cfg.feature_extractor_type,
        distance_metric=cfg.distance_metric,
        n_neighbors=cfg.n_neighbors,
        seed=cfg.seed,
        device=cfg.device,
    )

    encoder, y_pred = encode(dataset)

    save_stimuli(
        low_std=cfg.pos_std_threshold,
        high_std=np.inf,
        num_stimuli=cfg.num_stimuli,
        targets=y_pred,
        dataset=dataset,
        save_dir=max_dir,
    )

    perform_lora(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        lora_rank=cfg.rank,
        train_text_encoder=cfg.train_text_encoder,
        instance_prompt=cfg.instance_prompt,
        validation_prompt=cfg.validation_prompt,
        num_validation_images=cfg.num_validation_images,
        validation_epochs=cfg.validation_epochs,
        max_train_epochs=cfg.max_train_epochs,
        inference_steps=cfg.inference_steps,
        instance_data_root=max_dir,
        output_dir=lora_dir,
        resolution=cfg.resolution,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device=cfg.device,
    )

    feature_extractor = create_feature_extractor(
        type=cfg.feature_extractor_type,
        device=cfg.device,
    )
    predict_from_dir(
        encoder=encoder,
        feature_extractor=feature_extractor,
        preds_dir=lora_dir,
    )
