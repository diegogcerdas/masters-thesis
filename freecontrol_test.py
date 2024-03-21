import torch
from PIL import Image
import os
from freecontrol_pipeline import FreeControlSDPipeline

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
img_path = 'obama.png'
recon_path = 'recon.png'
output_path = 'freecontrol_output'
prompt = ""
num_inference_steps = 50
guidance_scale = 1
num_save_basis = 64
num_save_steps = 50
seed = 42

pipeline = FreeControlSDPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
img = Image.open(img_path).convert('RGB').resize((768, 768))
generator = torch.Generator(device=device).manual_seed(seed)

inverted_latents, prompt_embeds = pipeline.ddim_inversion(
    img,
    prompt,
    num_inference_steps,
    guidance_scale,
    generator,
)

recon = pipeline.ddim_sample(
    inverted_latents,
    num_inference_steps,
    prompt_embeds,
    num_save_basis,
    num_save_steps,
)

recon.save(recon_path)

os.makedirs(output_path, exist_ok=True)
torch.save(pipeline.pca_info, f"{output_path}/pca_info.pt")