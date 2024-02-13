import argparse
import os

import pytorch_lightning as pl
import torch

from model.brain_encoder import EncoderModule
from model.diffusion.gradient_descent import GradientDescent
from model.diffusion.stable_diffusion import StableDiffusion
from model.feature_extractor import create_feature_extractor
from utils.configs import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--brain-encoder-ckpt", type=str)
    parser.add_argument("--brain-encoder-desc", type=str)
    parser.add_argument("--feature-extractor-type", type=str, default="clip")

    # Synthesis Parameters
    parser.add_argument("--prompt", type=str, default="a photograph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-scale", type=float, default=1)
    parser.add_argument("--g", type=float, default=7.5)
    parser.add_argument("--inference-steps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/")
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="guided_synthesis")
    pl.seed_everything(cfg.seed)

    cfg.outputs_dir = os.path.join(cfg.outputs_dir, cfg.brain_encoder_desc)

    feature_extractor = create_feature_extractor(cfg.feature_extractor_type, cfg.device)
    encoder = EncoderModule.load_from_checkpoint(
        cfg.brain_encoder_ckpt, feature_extractor=feature_extractor
    )
    ldm = StableDiffusion(batch_size=1, device=cfg.device)

    # Initialize gradient descent
    gradient_descent = GradientDescent(
        ldm=ldm,
        brain_encoder=encoder,
        condition=ldm.text_enc([cfg.prompt]),
        loss_scale=cfg.loss_scale,
        g=cfg.g,
        seed=cfg.seed,
        steps=cfg.inference_steps,
        learning_rate=cfg.learning_rate,
    )

    # Create directory for outputs
    outputs_dir_prompt = os.path.join(
        cfg.outputs_dir, cfg.prompt.replace(" ", "_").strip()
    )
    outputs_dir_images = os.path.join(outputs_dir_prompt, "images")
    os.makedirs(outputs_dir_images, exist_ok=True)

    score_list = []
    for i in range(int(cfg.iterations)):
        # Zero the gradients
        gradient_descent.optimizer.zero_grad()

        # Compute score for current prompt
        score, latents = gradient_descent.forward()

        # Append score and save image
        with torch.no_grad():
            score_list.append(score.item() / cfg.loss_scale)
            pil_image = ldm.latents_to_image(latents, return_pil=True)[0]
            filename = f"{i}_{round(score.item()/cfg.loss_scale, 4)}.jpg"
            pil_image.save(os.path.join(outputs_dir_images, filename))

        # Gradient descent step
        loss = -score
        loss.backward()
        gradient_descent.optimizer.step()

    # Save scores to file
    with open(os.path.join(outputs_dir_prompt, "scores.txt"), "w") as file:
        for item in score_list:
            file.write(str(item) + "\n")
