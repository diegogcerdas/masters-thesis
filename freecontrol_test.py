import torch
from PIL import Image
from freecontrol_pipeline import FreeControlSDPipeline

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
img_path = 'obama.png'
recon_path = 'recon.png'
prompt = "a photo of obama"
num_inference_steps = 200
guidance_scale = 1
seed = 42

pipeline = FreeControlSDPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
img = Image.open(img_path).resize((512, 512))
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
)

recon.save(recon_path)

pipeline = FreeControlSDPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
img = Image.open(img_path)
generator = torch.Generator(device=device).manual_seed(seed)

inverted_latents, prompt_embeds = pipeline.ddim_inversion(
    img,
    "",
    num_inference_steps,
    guidance_scale,
    generator,
)

recon = pipeline.ddim_sample(
    inverted_latents,
    num_inference_steps,
    prompt_embeds,
)

recon.save('uncond.png')