import argparse
import os

import pytorch_lightning as pl
import torch
from diffusers import DiffusionPipeline

from models.stable_diffusion import StableDiffusion
from utils.configs import config_from_args
from utils.img_utils import save_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Synthesis Parameters
    parser.add_argument(
        "--prompt", type=str, default="an oil painting of a train station"
    )
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=500)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/vanilla/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="synthesis")
    pl.seed_everything(cfg.seed)

    ldm = StableDiffusion(
        cfg.pretrained_model_name_or_path,
        cfg.device,
    )
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
    )
    pipeline_args = {
        "prompt": cfg.prompt,
        "num_inference_steps": cfg.inference_steps,
    }
    images = ldm.sample(
        cfg.num_images,
        pipeline,
        pipeline_args,
        cfg.device,
        cfg.seed,
    )
    outputs_dir_prompt = os.path.join(
        cfg.outputs_dir, cfg.prompt.replace(" ", "_").strip()
    )
    save_images(images, outputs_dir_prompt)
