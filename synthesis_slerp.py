import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch

from model.stable_diffusion import StableDiffusion
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

    # Synthesis Parameters
    parser.add_argument(
        "--prompt1",
        type=str,
        default="an photograph of a cooking pan with delicious food",
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default="an photograph of a two persons eating food in a kitchen",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g", type=float, default=7.5)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/vanilla/")
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="synthesis2")
    pl.seed_everything(cfg.seed)

    ldm = StableDiffusion(batch_size=cfg.batch_size, device=cfg.device)

    imgs = []
    steps = np.linspace(0, 1, cfg.steps)
    for step in steps:
        condition1 = ldm.text_enc([cfg.prompt1])
        condition2 = ldm.text_enc([cfg.prompt2])
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

    # Create directory for outputs
    outputs_dir_prompt = os.path.join(
        cfg.outputs_dir, cfg.prompt.replace(" ", "_").strip()
    )
    outputs_dir_images = os.path.join(outputs_dir_prompt, "images")
    os.makedirs(outputs_dir_images, exist_ok=True)

    # Save images
    for i, img in enumerate(imgs):
        img.save(os.path.join(outputs_dir_images, f"{i}.png"))
