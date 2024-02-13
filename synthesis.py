import argparse
import os

import pytorch_lightning as pl
import torch

from model.diffusion.stable_diffusion import StableDiffusion
from utils.configs import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Synthesis Parameters
    parser.add_argument(
        "--prompt", type=str, default="a white big ball with white background"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g", type=float, default=7.5)
    parser.add_argument("--inference-steps", type=int, default=500)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/vanilla/")
    parser.add_argument("--batch-size", type=int, default=1)
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

    ldm = StableDiffusion(batch_size=cfg.batch_size, device=cfg.device)

    # Obtain text embeddings
    condition = ldm.text_enc([cfg.prompt] * cfg.batch_size)
    max_length = condition.shape[1]
    uncondition = ldm.text_enc([""] * cfg.batch_size, max_length)
    text_embedding = torch.cat([uncondition, condition])

    # Synthesize images
    imgs = ldm.text_emb_to_img(
        text_embedding=text_embedding,
        return_pil=True,
        g=cfg.g,
        seed=cfg.seed,
        steps=cfg.inference_steps,
    )

    # Create directory for outputs
    outputs_dir_prompt = os.path.join(
        cfg.outputs_dir, cfg.prompt.replace(" ", "_").strip()
    )
    outputs_dir_images = os.path.join(outputs_dir_prompt, "images")
    os.makedirs(outputs_dir_images, exist_ok=True)

    # Save images
    for i, img in enumerate(imgs):
        img.save(os.path.join(outputs_dir_images, f"{i}.png"))
