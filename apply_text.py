from diffusers import StableUnCLIPImg2ImgPipeline, StableDiffusionPipeline, DDIMInverseScheduler
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np
import torch
import numpy as np
from PIL import Image
from utils.img_utils import save_images
import open_clip

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed = 0

img_f = 'ant.png'
output_dir = './outputs/ant'
mults = np.arange(-5, 6, 1)
prompt = 'a photo of an ant'

word1 = 'big'
word2 = 'small'

@torch.no_grad()
def ddim_inversion(
    image: Image.Image,
    num_inference_steps: int,
    prompt: str,
    guidance_scale: float,
    seed: int,
    device: str,
):
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    # 1. Define call parameters
    do_classifier_free_guidance = guidance_scale > 1.0
    inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

    # 2. Preprocess image
    image = pipeline.image_processor.preprocess(image)

    def prepare_image_latents(pipeline, image, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        latents = pipeline.vae.encode(image).latent_dist.sample(generator)
        latents = pipeline.vae.config.scaling_factor * latents
        latents = torch.cat([latents], dim=0)
        return latents

    # 3. Prepare latent variables
    latents = prepare_image_latents(image, pipeline.vae.dtype, device, generator)

    # 4. Encode input prompt
    num_images_per_prompt = 1
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    )

    # 6. Prepare timesteps
    inverse_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = inverse_scheduler.timesteps

    # 7. Denoising loop
    for t in timesteps:

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample

    inverted_latents = latents.detach().clone()

    return inverted_latents, prompt_embeds

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip").to(device)
dtype = next(pipe.image_encoder.parameters()).dtype

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

img_test = Image.open(img_f).convert('RGB').resize((768,768))

text = tokenizer([word1, word2])
x = clip.encode_text(text)
shift_vector = x[0] - x[1]

inverted_latents, _ = ddim_inversion(
    image=img_test,
    num_inference_steps=50,
    prompt=prompt,
    guidance_scale=7.5,
    seed=seed,
    device=device,
)

# Get CLIP vision embeddings
img_test = pipe.feature_extractor(images=img_test, return_tensors="pt").pixel_values
img_test = img_test.to(device=device, dtype=dtype)
img_test_embeds = pipe.image_encoder(img_test).image_embeds

images = []
for i, mult in enumerate(mults):
    generator = torch.Generator(device=device).manual_seed(seed)
    emb = img_test_embeds + mult * shift_vector
    img = pipe(
        latents=inverted_latents,
        prompt=prompt,
        generator=generator,
        image_embeds=emb,
        noise_level=0
    ).images[0]
    images.append(img)

save_dir = f'{output_dir}/{word1}_{word2}'
save_images(images, save_dir)
