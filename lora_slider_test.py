import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel
from lora_slider_utils import LoRANetwork
from tqdm import tqdm
from utils.img_utils import save_images

def run_test(
    args_pretrained_model_name_or_path: str,
    args_alpha: float, 
    args_rank: int,  
    args_training_method: str,
    args_lora_weights_path: str,
    args_prompt: str,
    args_scales: list,
    args_start_noise: int,
    args_seed: int,
    args_device: str,
    args_save_folder: str,
):
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5

    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="text_encoder").to(args_device)
    vae = AutoencoderKL.from_pretrained(args_pretrained_model_name_or_path, subfolder="vae").to(args_device)
    unet = UNet2DConditionModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="unet").to(args_device)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    noise_scheduler.set_timesteps(ddim_steps)

    network = LoRANetwork(
        unet,
        rank=args_rank,
        alpha=args_alpha,
        train_method=args_training_method,
    ).to(args_device)
    network.load_state_dict(torch.load(args_lora_weights_path))
    
    images_list = []

    for scale in args_scales:

        generator = torch.manual_seed(args_seed) 

        cond_input = tokenizer(args_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = text_encoder(cond_input.input_ids.to(args_device))[0]
        max_length = cond_input.input_ids.shape[-1]
        uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(args_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        latents = torch.randn((1, unet.in_channels, height // 8, width // 8), generator=generator, device=args_device)
        latents = latents * noise_scheduler.init_noise_sigma
        
        for t in tqdm(noise_scheduler.timesteps):

            if t > args_start_noise:
                network.set_lora_slider(scale=0)
            else:
                network.set_lora_slider(scale=scale)

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            with network:
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        images_list.append(pil_images[0])

    save_images(images_list, args_save_folder)