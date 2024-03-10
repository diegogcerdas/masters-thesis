# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

import torch
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPTextModel

from lora_slider_utils import (
    load_prompts_from_yaml, 
    encode_prompts, 
    PromptEmbedsPair,
    LoRANetwork
)

def prev_step(model_output, timestep, scheduler, sample):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t =scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample

def run(
    args_pretrained_model_name_or_path: str,  # Name or path of the pretrained model
    args_alpha: float,  # LoRA weight
    args_rank: int,  # Rank of LoRA
    args_training_method: str,  # Training method of LoRA
    args_prompt_file: str,  # Path to the prompt file
    args_learning_rate: float,  # Learning rate
    args_seed: int,  # Seed
    args_device: str,  # Device to use
):
    
    prompt_settings = load_prompts_from_yaml(args_prompt_file)
    scales = np.array(scales)
    folders = np.array(folders)
    scales_unique = list(scales)

    noise_scheduler = DDPMScheduler.from_pretrained(args_pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="text_encoder").to(args_device)
    vae = AutoencoderKL.from_pretrained(args_pretrained_model_name_or_path, subfolder="vae").to(args_device)
    unet = UNet2DConditionModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="unet").to(args_device)

    text_encoder.eval()
    unet.requires_grad_(False)
    unet.eval()
    vae.requires_grad_(False)
    vae.eval()

    network = LoRANetwork(
        unet,
        rank=args_rank,
        alpha=args_alpha,
        train_method=args_training_method,
    ).to(args_device)
            
    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=args_learning_rate)
    criteria = torch.nn.MSELoss()

    with torch.no_grad():
        prompt_pair = PromptEmbedsPair(
            loss_fn=torch.nn.MSELoss(),
            target=encode_prompts(tokenizer, text_encoder, [prompt_settings.target]),
            positive=encode_prompts(tokenizer, text_encoder, [prompt_settings.positive]),
            unconditional=encode_prompts(tokenizer, text_encoder, [prompt_settings.neutral]),
            neutral=encode_prompts(tokenizer, text_encoder, [prompt_settings.unconditional]),
            settings=prompt_settings,
        )

    pbar = tqdm(range(config.train.iterations))
    for i in pbar:

        with torch.no_grad():
            noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=args_device)
            optimizer.zero_grad()
            timesteps_to = torch.randint(1, config.train.max_denoising_steps-1, (1,)).item()

            scale_to_look = abs(random.choice(list(scales_unique)))
            folder1 = folders[scales==-scale_to_look][0]
            folder2 = folders[scales==scale_to_look][0]
            
            ims = os.listdir(f'{folder_main}/{folder1}/')
            ims = [im_ for im_ in ims if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
            random_sampler = random.randint(0, len(ims)-1)

            img1 = Image.open(f'{folder_main}/{folder1}/{ims[random_sampler]}').resize((256,256))
            img2 = Image.open(f'{folder_main}/{folder2}/{ims[random_sampler]}').resize((256,256))
            
            generator = torch.manual_seed(args_seed)
            denoised_latents_low, low_noise = train_util.get_noisy_image(
                img1,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_low = denoised_latents_low.to(args_device)
            low_noise = low_noise.to(args_device)
            
            generator = torch.manual_seed(args_seed)
            denoised_latents_high, high_noise = train_util.get_noisy_image(
                img2,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_high = denoised_latents_high.to(args_device)
            high_noise = high_noise.to(args_device)
            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]


            high_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            low_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
        
        network.set_lora_slider(scale=scale_to_look)
        with network:
            target_latents_high = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
            
        high_latents.requires_grad = False
        low_latents.requires_grad = False
        
        loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_high.item()*1000:.4f}")
        loss_high.backward()
        
        
        network.set_lora_slider(scale=-scale_to_look)
        with network:
            target_latents_low = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
            
        high_latents.requires_grad = False
        low_latents.requires_grad = False
        
        loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_low.item()*1000:.4f}")
        loss_low.backward()
        
        ## NOTICE NO zero_grad between these steps (accumulating gradients) 
        # following guidelines from Ostris (https://github.com/ostris/ai-toolkit)
        
        optimizer.step()

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )