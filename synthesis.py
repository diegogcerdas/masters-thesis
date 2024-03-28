import argparse
import os

import pytorch_lightning as pl
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from utils.configs import config_from_args
from utils.img_utils import save_images
from utils.configs import ConfigSynthesis

def main(cfg: ConfigSynthesis):
    
    # set seed
    pl.seed_everything(cfg.seed)

    # load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(cfg.pretrained_model_name_or_path).to(cfg.device)

    # disable safety checker
    pipeline.safety_checker = None   

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(cfg.inference_steps)

    # optionally load LoRA weights
    if cfg.lora_dir is not None:
        pipeline.load_lora_weights(cfg.lora_dir, weight_name="pytorch_lora_weights.safetensors")
    
    # run the pipeline
    pipeline_args = {
        "prompt": cfg.prompt,
        "num_inference_steps": cfg.inference_steps,
    }
    images = []
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.device)
    for _ in range(cfg.num_images):
        with torch.cuda.amp.autocast():
            image = pipeline(**pipeline_args, generator=generator).images[0]
            images.append(image)

    # save the images
    outputs_dir_prompt = os.path.join(
        cfg.outputs_dir, cfg.prompt.replace(" ", "_").strip()
    )
    save_images(images, outputs_dir_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained-model-name-or-path", type=str, default="stabilityai/stable-diffusion-2",)
    parser.add_argument("--prompt", type=str, default="an picture of a cat with background")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/synthesis/")
    parser.add_argument("--lora-dir", type=str, default=None)
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
    main(cfg)