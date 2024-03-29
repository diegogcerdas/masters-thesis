import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from pytorch_lightning import seed_everything
from methods.diffmorpher_utils import AlphaScheduler


def interpolate(
    pretrained_model_name_or_path: str,
    lora_min_path: str,
    lora_max_path: str,
    prompt: str,
    num_timesteps: int,
    num_frames: int,
    seed: int,
    device: str,
):
    seed_everything(seed)
    
    # Load the pretrained model
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)

    # Disable safety checker
    pipe.safety_checker = None  

    # Load LoRA weights
    pipe.load_lora_weights(
        lora_min_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="lora_min",
    )
    pipe.load_lora_weights(
        lora_max_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="lora_max",
    )

    # Switch to DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_timesteps)

    def morph(alpha_list, desc):
        images = []
        for alpha in tqdm(alpha_list, desc=desc):
            # Set the adapter weights
            pipe.set_adapters(["lora_min", "lora_max"], adapter_weights=[1 - alpha, alpha])
            # Run the pipeline  
            generator = torch.Generator(device=device).manual_seed(seed)        
            with torch.cuda.amp.autocast():
                image = pipe(
                    num_inference_steps=num_timesteps,
                    prompt=prompt,
                    generator=generator,
                ).images[0]
            images.append(image)
        return images

    with torch.no_grad():
        
        alpha_list = list(torch.linspace(0, 1, num_frames))
        images_pt = [transforms.ToTensor()(img).unsqueeze(0) for img in morph(alpha_list, "Sampling...")]
        
        alpha_list = AlphaScheduler(images_pt).get_list()
        images = morph(alpha_list, "Reschedule...")

    return images, alpha_list
