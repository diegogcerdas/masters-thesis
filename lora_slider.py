import os
import random

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from lora_slider_utils import (LoRANetwork, concat_embeddings, encode_prompts,
                               get_noisy_image, predict_noise)


def run_train(
    args_pretrained_model_name_or_path: str,  # Name or path of the pretrained model
    args_alpha: float,  # LoRA weight
    args_rank: int,  # Rank of LoRA
    args_training_method: str,  # Training method of LoRA
    args_train_steps: int,  # Number of training steps
    args_max_denoising_steps: int,  # Maximum denoising steps
    args_folder_main: str,  # Path to the main folder
    args_learning_rate: float,  # Learning rate
    args_seed: int,  # Seed
    args_device: str,  # Device to use
    args_save_path: str,  # Path to save the model
):
    scales = np.array([0, 5])
    folders = np.array(["null", "max"])

    noise_scheduler = DDPMScheduler.from_pretrained(
        args_pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args_pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args_pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(args_device)
    vae = AutoencoderKL.from_pretrained(
        args_pretrained_model_name_or_path, subfolder="vae"
    ).to(args_device)
    unet = UNet2DConditionModel.from_pretrained(
        args_pretrained_model_name_or_path, subfolder="unet"
    ).to(args_device)

    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.eval()
    vae.eval()

    network = LoRANetwork(
        unet,
        rank=args_rank,
        alpha=args_alpha,
        train_method=args_training_method,
    ).to(args_device)

    optimizer = torch.optim.AdamW(
        network.prepare_optimizer_params(), lr=args_learning_rate
    )
    criteria = torch.nn.MSELoss()

    # prompts
    unconditional_prompt = encode_prompts(tokenizer, text_encoder, [""])
    positive_prompt = encode_prompts(tokenizer, text_encoder, [""])
    neutral_prompt = encode_prompts(tokenizer, text_encoder, [""])

    for i in tqdm(range(args_train_steps)):
        with torch.no_grad():
            optimizer.zero_grad()

            noise_scheduler.set_timesteps(args_max_denoising_steps, device=args_device)
            timesteps_to = torch.randint(1, args_max_denoising_steps - 1, (1,)).item()

            img_fs = os.listdir(f"{args_folder_main}/{folders[0]}/")
            img_fs = [
                im_
                for im_ in img_fs
                if ".png" in im_ or ".jpg" in im_ or ".jpeg" in im_ or ".webp" in im_
            ]
            img_f = img_fs[random.randint(0, len(img_fs) - 1)]

            img1 = Image.open(f"{args_folder_main}/{folders[0]}/{img_f}").resize(
                (512, 512)
            )
            img2 = Image.open(f"{args_folder_main}/{folders[1]}/{img_f}").resize(
                (512, 512)
            )

            generator = torch.Generator(device=args_device).manual_seed(args_seed)
            noisy_latents_low, noise_low = get_noisy_image(
                img1,
                vae,
                generator,
                noise_scheduler,
                timesteps_to,
            )
            noisy_latents_low = noisy_latents_low.to(args_device)
            noise_low = noise_low.to(args_device)

            generator = torch.Generator(device=args_device).manual_seed(args_seed)
            noisy_latents_high, noise_high = get_noisy_image(
                img2,
                vae,
                generator,
                noise_scheduler,
                timesteps_to,
            )
            noisy_latents_high = noisy_latents_high.to(args_device)
            noise_high = noise_high.to(args_device)

            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / args_max_denoising_steps)
            ]

        network.set_lora_slider(scale=scales[0])
        with network:
            noise_pred_low = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                noisy_latents_low,
                concat_embeddings(
                    unconditional_prompt,
                    neutral_prompt,
                ),
                guidance_scale=1,
            )

        loss_low = criteria(noise_pred_low, noise_low)
        print(f"{i} Loss*1k low: {loss_low.item()*1000:.4f}")
        loss_low.backward()

        network.set_lora_slider(scale=scales[1])
        with network:
            noise_pred_high = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                noisy_latents_high,
                concat_embeddings(
                    unconditional_prompt,
                    positive_prompt,
                ),
                guidance_scale=1,
            )

        loss_high = criteria(noise_pred_high, noise_high)
        print(f"{i} Loss*1k high: {loss_high.item()*1000:.4f}")
        loss_high.backward()

        optimizer.step()

    print("Saving...")
    os.makedirs(args_save_path, exist_ok=True)
    network.save_weights(os.path.join(args_save_path, "last.pt"))
