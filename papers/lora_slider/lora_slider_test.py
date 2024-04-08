import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from lora_slider_utils import LoRANetwork
from utils.img_utils import save_images


def run_test(
    pretrained_model_name_or_path: str,
    alpha: float,
    rank: int,
    training_method: str,
    lora_weights_path: str,
    prompt: str,
    scales: list,
    start_noise: int,
    seed: int,
    device: str,
    save_folder: str,
):
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5

    # Load the pretrained model
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Disable safety checker
    pipe.safety_checker = None   

    # Switch to DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_timesteps)

    network = LoRANetwork(
        pipe.unet,
        rank=rank,
        alpha=alpha,
        train_method=training_method,
    ).to(device)
    network.load_state_dict(torch.load(lora_weights_path))

    images_list = []

    for scale in scales:

        generator = torch.Generator(device=device).manual_seed(seed)

        cond_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        cond_embeddings = text_encoder(cond_input.input_ids.to(device))[0]
        max_length = cond_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        latents = torch.randn(
            (1, unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
        )
        latents = latents * noise_scheduler.init_noise_sigma

        for t in tqdm(noise_scheduler.timesteps):
            if t > start_noise:
                network.set_lora_slider(scale=0)
            else:
                network.set_lora_slider(scale=scale)

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            with network:
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        images_list.append(pil_images[0])

    save_images(images_list, save_folder)
