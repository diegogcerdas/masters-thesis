import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
from PIL import Image


@torch.no_grad()
def ddim_inversion(
    pretrained_model_name_or_path: str,
    image: Image.Image,
    num_inference_steps: int,
    prompt: str,
    guidance_scale: float,
    seed: int,
    device: str,
):
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
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
    latents = prepare_image_latents(pipeline, image, pipeline.vae.dtype, device, generator)

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