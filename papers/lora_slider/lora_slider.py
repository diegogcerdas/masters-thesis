import os
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, LMSDiscreteScheduler
from pytorch_lightning import seed_everything
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from utils.img_utils import save_images

from lora_slider_utils import (LoRANetwork, concat_embeddings, encode_prompts,
                               get_noisy_image, predict_noise)


def run_train(
    pretrained_model_name_or_path: str,
    resolution: int,
    alpha: float,
    rank: int, 
    scales: np.ndarray,
    folders: np.ndarray,
    training_method: str, 
    num_timesteps: int, 
    folder_main: str, 
    save_path: str,
    num_epochs: int,  
    learning_rate: float,
    validation_scales: list,
    validation_prompt: str,
    validation_guidance_scale: float,
    validation_start_noise: int,
    validation_epochs: int,
    seed: int,  
    device: str, 
):
    seed_everything(seed)

    # Load the pretrained model
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()
    pipe.vae.eval()

    # Disable safety checker
    pipe.safety_checker = None   

    # Switch to DDIM scheduler
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    network = LoRANetwork(
        pipe.unet,
        rank=rank,
        alpha=alpha,
        train_method=training_method,
    ).to(device)

    optimizer = torch.optim.AdamW(
        network.prepare_optimizer_params(), lr=learning_rate
    )

    # prompts
    unconditional_prompt = encode_prompts(pipe.tokenizer, pipe.text_encoder, [""])
    positive_prompt = encode_prompts(pipe.tokenizer, pipe.text_encoder, [""])
    neutral_prompt = encode_prompts(pipe.tokenizer, pipe.text_encoder, [""])

    filenames = np.array([
        f for f in os.listdir(os.path.join(folder_main, folders[0])) if f.endswith(".png")
    ])

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        ################### VALIDATION ###################

        if epoch % validation_epochs == 0:

            pipe.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

            images_list = []

            for scale in validation_scales:

                pipe.scheduler.set_timesteps(num_timesteps, device=device)

                generator = torch.Generator(device=device).manual_seed(seed)

                cond_input = pipe.tokenizer(
                    validation_prompt,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                cond_embeddings = pipe.text_encoder(cond_input.input_ids.to(device))[0]
                max_length = cond_input.input_ids.shape[-1]
                uncond_input = pipe.tokenizer(
                    [""], padding="max_length", max_length=max_length, return_tensors="pt"
                )
                uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
                text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

                latents = torch.randn(
                    (1, pipe.unet.in_channels, resolution // 8, resolution // 8),
                    generator=generator,
                    device=device,
                )
                latents = latents * pipe.scheduler.init_noise_sigma

                for t in tqdm(pipe.scheduler.timesteps):
                    if t > validation_start_noise:
                        network.set_lora_slider(scale=0)
                    else:
                        network.set_lora_slider(scale=scale)

                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = pipe.scheduler.scale_model_input(
                        latent_model_input, timestep=t
                    )

                    with network:
                        with torch.no_grad():
                            noise_pred = pipe.unet(
                                latent_model_input, t, encoder_hidden_states=text_embeddings
                            ).sample

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + validation_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = pipe.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                images_list.append(pil_images[0])

            save_path_epoch = os.path.join(save_path, f"epoch_{epoch}")
            save_images(images_list, save_path_epoch)
            network.save_weights(os.path.join(save_path_epoch, "ckpt.pt"))

        #################### TRAINING ####################

        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        filenames = filenames[np.random.permutation(len(filenames))]

        for f in filenames:

            pipe.scheduler.set_timesteps(num_timesteps, device=device)
            timestep = torch.randint(1, num_timesteps - 1, (1,)).item()

            img1 = Image.open(f"{folder_main}/{folders[0]}/{f}").resize((resolution, resolution))
            img2 = Image.open(f"{folder_main}/{folders[1]}/{f}").resize((resolution, resolution))
        
            with torch.no_grad():

                generator = torch.Generator(device=device).manual_seed(seed)
                noisy_latents_low, noise_low = get_noisy_image(
                    img1,
                    pipe.vae,
                    generator,
                    pipe.scheduler,
                    timestep,
                )
                noisy_latents_low = noisy_latents_low.to(device)
                noise_low = noise_low.to(device)

                generator = torch.Generator(device=device).manual_seed(seed)
                noisy_latents_high, noise_high = get_noisy_image(
                    img2,
                    pipe.vae,
                    generator,
                    pipe.scheduler,
                    timestep,
                )
                noisy_latents_high = noisy_latents_high.to(device)
                noise_high = noise_high.to(device)

                pipe.scheduler.set_timesteps(1000)
                current_timestep = pipe.scheduler.timesteps[
                    int(timestep * 1000 / num_timesteps)
                ]

            network.set_lora_slider(scale=scales[0])
            with network:
                noise_pred_low = predict_noise(
                    pipe.unet,
                    pipe.scheduler,
                    current_timestep,
                    noisy_latents_low,
                    concat_embeddings(
                        unconditional_prompt,
                        neutral_prompt,
                    ),
                    guidance_scale=1,
                )

            loss_low = F.mse_loss(noise_pred_low, noise_low)
            loss_low.backward()

            network.set_lora_slider(scale=scales[1])
            with network:
                noise_pred_high = predict_noise(
                    pipe.unet,
                    pipe.scheduler,
                    current_timestep,
                    noisy_latents_high,
                    concat_embeddings(
                        unconditional_prompt,
                        positive_prompt,
                    ),
                    guidance_scale=1,
                )

            loss_high = F.mse_loss(noise_pred_high, noise_high)
            loss_high.backward()

            optimizer.step()
            optimizer.zero_grad()
