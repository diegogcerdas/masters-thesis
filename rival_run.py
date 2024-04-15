import json
import os
import torch
import types
from diffusers import DDIMScheduler
from rival.ddim_inversion import Inversion, load_512
from diffusers.models.attention import Attention
from rival.sd_pipeline_img import RIVALStableDiffusionPipeline
from rival.attention_forward import new_forward


def run(
    pretrained_model_name_or_path: str,
    num_ddim_steps: int,
    invert_steps: int,
    guidance_scale: float,
    t_early: int,
    t_align: int,
    atten_frames: int,
    image_path: str,
    prompt_original: str,
    prompt_edit: str,
    num_images: int,
    editing_early_steps: int,
    outputs_dir: str,
    seed: int,
    device: str,
):
    
    ldm_stable = RIVALStableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
    ldm_stable.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)

    for module in ldm_stable.unet.modules():
        if isinstance(module, Attention):
            # use a placeholder function for the original forward.
            module.ori_forward = module.forward
            module.cfg = {
                "atten_frames": atten_frames,
                "t_align": t_align      
            }
            module.init_step = 1000
            module.step_size = module.init_step // num_ddim_steps
            module.t_align = module.cfg["t_align"]
            module.editing_early_steps = editing_early_steps
            module.forward = types.MethodType(new_forward, module)
    
    prompts = [prompt_original, prompt_edit]
    generator = torch.Generator(device=device).manual_seed(seed)

    inversion = Inversion(ldm_stable, guidance_scale, num_ddim_steps, invert_steps)
    inversion.init_prompt(prompt_original)  
    image_gt = load_512(image_path)
    _, x_ts = inversion.ddim_inversion(image_gt)
    x_t_in = torch.cat([x_ts[-1], x_ts[-1]], dim=0)
        
    for m in range(num_images):
        
        with torch.no_grad():
            image = ldm_stable(
                prompts,
                generator=generator,
                latents=x_t_in,
                num_images_per_prompt=1,
                num_inference_steps = num_ddim_steps,
                guidance_scale = guidance_scale,
                is_adain = True,
                chain = x_ts,
                t_early = t_early,
                output_type = 'pil',
            ).images[0]

        os.makedirs(outputs_dir, exist_ok=True)
        image.save(os.path.join(outputs_dir, f'{m}.png'))     