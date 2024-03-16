import torch
import tqdm
import safetensors
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline

from diffmorpher_utils import StoreProcessor, LoadProcessor, get_text_embeddings, image2latent, load_lora, ddim_inversion, cal_image, AlphaScheduler

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
    args_attn_beta: float,
    args_lamd: float,
    args_use_adain: bool,
    args_use_reschedule: bool,
    args_resolution: int,
    args_device: str,
):
    pipe = StableDiffusionPipeline.from_pretrained(args_pretrained_model_name_or_path)
    pipe.scheduler.set_timesteps(args_num_inference_steps)

    lora_0 = safetensors.torch.load_file(args_load_lora_path_0, device="cpu")
    lora_1 = safetensors.torch.load_file(args_load_lora_path_1, device="cpu")

    text_embeddings_0 = get_text_embeddings(pipe.tokenizer, pipe.text_encoder, args_prompt_0)
    text_embeddings_1 = get_text_embeddings(pipe.tokenizer, pipe.text_encoder, args_prompt_1)
    
    img_0 = Image.open(args_img_path_0)
    img_1 = Image.open(args_img_path_1)
    img_0 = image2latent(pipe.vae, img_0, args_resolution, args_device)
    img_1 = image2latent(pipe.vae, img_1, args_resolution, args_device)
    
    pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 0)
    img_noise_0 = ddim_inversion(pipe.unet, pipe.scheduler, img_0, text_embeddings_0)
    pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 1)
    img_noise_1 = ddim_inversion(pipe.unet, pipe.scheduler, img_1, text_embeddings_1)

    def morph(alpha_list, desc):
        
        img0_dict = dict()
        img1_dict = dict()
    
        images = []

        if args_attn_beta is not None:

            pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 0)

            attn_processor_dict = {}
            for k in pipe.unet.attn_processors.keys():
                if k.startswith('up'):
                    attn_processor_dict[k] = StoreProcessor(pipe.unet.attn_processors[k], img0_dict, k)
                else:
                    attn_processor_dict[k] = pipe.unet.attn_processors[k]
            pipe.unet.set_attn_processor(attn_processor_dict)

            first_image = cal_image(
                args_num_inference_steps,
                img_noise_0,
                img_noise_1,
                text_embeddings_0,
                text_embeddings_1,
                alpha_list[0],
                args_use_adain,
            )

            pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 1)

            attn_processor_dict = {}
            for k in pipe.unet.attn_processors.keys():
                if k.startswith('up'):
                    attn_processor_dict[k] = StoreProcessor(pipe.unet.attn_processors[k], img1_dict, k)
                else:
                    attn_processor_dict[k] = pipe.unet.attn_processors[k]
            pipe.unet.set_attn_processor(attn_processor_dict)

            last_image = cal_image(
                args_num_inference_steps,
                img_noise_0,
                img_noise_1,
                text_embeddings_0,
                text_embeddings_1,
                alpha_list[-1],
                args_use_adain,
            )

            for i in tqdm(range(1, args_num_frames - 1), desc=desc):
                
                alpha = alpha_list[i]
                pipe.unet = load_lora(pipe.unet, lora_0, lora_1, alpha)

                attn_processor_dict = {}
                for k in pipe.unet.attn_processors.keys():
                    if k.startswith('up'):
                        attn_processor_dict[k] = LoadProcessor(pipe.unet.attn_processors[k], k, img0_dict, img1_dict, alpha, args_attn_beta, args_lamd)
                    else:
                        attn_processor_dict[k] = pipe.unet.attn_processors[k]
                pipe.unet.set_attn_processor(attn_processor_dict)

                image = cal_image(
                    args_num_inference_steps,
                    img_noise_0,
                    img_noise_1,
                    text_embeddings_0,
                    text_embeddings_1,
                    alpha,
                    args_use_adain,
                )
                images.append(image)

            images = [first_image] + images + [last_image]

        else:
            for alpha in tqdm(alpha_list, desc=desc):

                pipe.unet = load_lora(pipe.unet, lora_0, lora_1, alpha)

                image = cal_image(
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