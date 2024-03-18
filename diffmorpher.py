import torch
import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms

from diffmorpher_utils import (AlphaScheduler, cal_image, ddim_inversion,
                               get_text_embeddings, image2latent)


def run(
    args_pretrained_model_name_or_path: str,
    args_img_path_0: str,
    args_img_path_1: str,
    args_load_lora_path_0: str,
    args_load_lora_path_1: str,
    args_prompt_0: str,
    args_prompt_1: str,
    args_num_inference_steps: int,
    args_num_frames: int,
    args_use_adain: bool,
    args_use_reschedule: bool,
    args_resolution: int,
    args_seed: int,
    args_device: str,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        args_pretrained_model_name_or_path
    ).to(args_device)

    pipe.load_lora_weights(
        args_load_lora_path_0,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="lora_0",
    )
    pipe.load_lora_weights(
        args_load_lora_path_1,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="lora_1",
    )

    text_embeddings_0 = get_text_embeddings(
        pipe.tokenizer, pipe.text_encoder, args_prompt_0, args_device
    )
    text_embeddings_1 = get_text_embeddings(
        pipe.tokenizer, pipe.text_encoder, args_prompt_1, args_device
    )

    img_0 = Image.open(args_img_path_0)
    img_1 = Image.open(args_img_path_1)
    img_0 = image2latent(pipe.vae, img_0, args_resolution, args_device)
    img_1 = image2latent(pipe.vae, img_1, args_resolution, args_device)

    pipe.scheduler.set_timesteps(args_num_inference_steps)
    alpha = 0
    pipe.set_adapters(["lora_0", "lora_1"], adapter_weights=[1 - alpha, alpha])
    img_noise_0 = ddim_inversion(pipe.unet, pipe.scheduler, img_0, text_embeddings_0)
    alpha = 1
    pipe.set_adapters(["lora_0", "lora_1"], adapter_weights=[1 - alpha, alpha])
    img_noise_1 = ddim_inversion(pipe.unet, pipe.scheduler, img_1, text_embeddings_1)

    generator = torch.Generator(device=args_device).manual_seed(args_seed)
    img_noise_0 = torch.randn(img_noise_0.shape, generator=generator, device=args_device)
    img_noise_1 = img_noise_0

    def morph(alpha_list, desc):
        images = []
        for alpha in tqdm(alpha_list, desc=desc):
            pipe.set_adapters(["lora_0", "lora_1"], adapter_weights=[1 - alpha, alpha])

            image = cal_image(
                pipe,
                args_num_inference_steps,
                img_noise_0,
                img_noise_1,
                text_embeddings_0,
                text_embeddings_1,
                alpha,
                args_use_adain,
            )
            images.append(image)

        return images

    with torch.no_grad():
        if args_use_reschedule:
            alpha_scheduler = AlphaScheduler()
            alpha_list = list(torch.linspace(0, 1, args_num_frames))
            images_pt = morph(alpha_list, "Sampling...")
            images_pt = [transforms.ToTensor()(img).unsqueeze(0) for img in images_pt]
            alpha_scheduler.from_imgs(images_pt)
            alpha_list = alpha_scheduler.get_list()
            images = morph(alpha_list, "Reschedule...")
        else:
            alpha_list = list(torch.linspace(0, 1, args_num_frames))
            images = morph(alpha_list, "Sampling...")

    return images, alpha_list
