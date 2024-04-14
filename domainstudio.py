import itertools
import os
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_lightning import seed_everything
import uuid
from utils.img_utils import save_images
from domainstudio_utils import DreamBoothDataset, collate_fn, get_wave

def run(
    pretrained_model_name_or_path: str,
    resolution: int,
    num_timesteps: int,
    instance_prompt: str,
    train_text_encoder: bool,
    class_prompt: str,
    instance_data_dir: str,
    class_data_dir: str,
    num_class_images: int,
    validation_epochs: int,
    num_validation_images: int,
    validation_prompt: str,
    prior_loss_weight: float,
    image_loss_weight: float,
    hf_loss_weight: float,
    hfmse_loss_weight: float,
    learning_rate: float,
    num_train_epochs: int,
    train_batch_size: int,
    outputs_dir: str,
    seed: int,
    device: str,
):
    seed_everything(seed)

    # Generate images for prior preservation
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    os.makedirs(class_data_dir, exist_ok=True)
    cur_class_images = len([f for f in os.listdir(class_data_dir) if f.endswith(".png")])

    if cur_class_images < num_class_images:

        num_new_images = num_class_images - cur_class_images
        images = []
        generator = torch.Generator(device=device).manual_seed(seed)
        for _ in tqdm(range(num_new_images), desc="Generating images"):
            with torch.cuda.amp.autocast():
                image = pipeline(
                    prompt=class_prompt,
                    num_inference_steps=num_timesteps,
                    generator=generator).images[0]
                images.append(image)

        names = [f'{str(uuid.uuid4())}.png' for _ in images]
        save_images(images, class_data_dir, names)
        del images

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)
    unet_source = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

    unet_source.requires_grad_(False)
    vae.requires_grad_(False)
    if not train_text_encoder:
        text_encoder.requires_grad_(False)

    if is_xformers_available():
        vae.enable_xformers_memory_efficient_attention()
        unet.enable_xformers_memory_efficient_attention()

    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters()
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    train_dataset = DreamBoothDataset(
        instance_data_dir=instance_data_dir,
        instance_prompt=instance_prompt,
        class_data_dir=class_data_dir,
        class_prompt=class_prompt,
        num_class_images=num_class_images,
        tokenizer=tokenizer,
        size=resolution,
    )
    
    c_fn = lambda examples: collate_fn(examples, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=c_fn, pin_memory=True
    )

    train_dataloader2 = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=c_fn, pin_memory=True
    )

    text_enc_context = nullcontext() if train_text_encoder else torch.no_grad()
    sfm = torch.nn.Softmax(dim=1)
    kl_loss = torch.nn.KLDivLoss()
    sim = torch.nn.CosineSimilarity()

    for epoch in range(num_train_epochs):

        #################### VALIDATION ####################

        if epoch % validation_epochs == 0:

            # Create the pipeline using using the trained modules and save it.
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                unet=unet,
                text_encoder=text_encoder,
                vae=vae,
                safety_checker=None,
            )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            save_dir = os.path.join(outputs_dir, str(epoch))
            pipeline.save_pretrained(save_dir)

            pipeline = pipeline.to(device)
            g_cuda = torch.Generator(device=device).manual_seed(seed)
            pipeline.set_progress_bar_config(disable=True)
            sample_dir = os.path.join(save_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            with torch.autocast("cuda"), torch.inference_mode():
                for i in tqdm(range(num_validation_images), desc="Generating samples"):
                    images = pipeline(
                        validation_prompt,
                        num_inference_steps=num_timesteps,
                        generator=g_cuda
                    ).images
                    images[0].save(os.path.join(sample_dir, f"{i}.png"))
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ##################### TRAINING #####################
        
        unet.train()
        if train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):

            # Convert images to latent space
            with torch.no_grad():
                latent_dist = vae.encode(batch["pixel_values"].to(device)).latent_dist
                latents = latent_dist.sample() * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Get the text embedding for conditioning
            with text_enc_context:
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            # Sample a random timestep for each image
            timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,))
            timesteps = (timestep[0] * torch.ones(bsz,)).to(latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            latents_pairwise = latents
            noise_pairwise = noise
            noisy_latents_pairwise = noise_scheduler.add_noise(latents_pairwise, noise_pairwise, timesteps)
            
            noisy_residual = unet(noisy_latents_pairwise, timesteps, encoder_hidden_states).sample
            pred_sample = noise_scheduler.step(noisy_residual, timestep[0], noisy_latents_pairwise).pred_original_sample

            pred_instance, pred_class = torch.chunk(pred_sample, 2, dim=0)

            model_pred_image = vae.decode(pred_instance.float()).sample
            model_pred_image_prior = vae.decode(pred_class.float()).sample

            bs_part = model_pred_image.shape[0]

            dist_source = torch.zeros([bs_part, bs_part-1]).cuda()
            for pair1 in range(bs_part):
                tmpc = 0
                anchor_feat = torch.unsqueeze(model_pred_image_prior[pair1].reshape(-1),0)
                for pair2 in range(bs_part):
                    if pair1 != pair2:
                        target_feat = torch.unsqueeze(model_pred_image_prior[pair2].reshape(-1),0)
                        dist_source[pair1, tmpc] = sim(anchor_feat, target_feat)
                        tmpc += 1
            dist_source = sfm(dist_source)

            dist_target = torch.zeros([bs_part, bs_part-1]).cuda()
            for pair1 in range(bs_part):
                tmpc = 0
                anchor_feat = torch.unsqueeze(model_pred_image[pair1].reshape(-1),0)
                for pair2 in range(bs_part):
                    if pair1 != pair2:
                        target_feat = torch.unsqueeze(model_pred_image[pair2].reshape(-1),0)
                        dist_target[pair1, tmpc] = sim(anchor_feat, target_feat)
                        tmpc += 1
            dist_target = sfm(dist_target)

            rel_loss = kl_loss(torch.log(dist_target), dist_source)

            LH, HL, HH = get_wave(model_pred_image)

            model_pred_image_wave = LH(model_pred_image)+HL(model_pred_image)+HH(model_pred_image)
            model_pred_prior_wave = LH(model_pred_image_prior)+HL(model_pred_image_prior)+HH(model_pred_image_prior)
            
            dist_source = torch.zeros([bs_part, bs_part-1]).cuda()
            for pair1 in range(bs_part):
                tmpc = 0
                anchor_feat = torch.unsqueeze(model_pred_image_wave[pair1].reshape(-1),0)
                for pair2 in range(bs_part):
                    if pair1 != pair2:
                        target_feat = torch.unsqueeze(model_pred_image_wave[pair2].reshape(-1),0)
                        dist_source[pair1, tmpc] = sim(anchor_feat, target_feat)
                        tmpc += 1
            dist_source = sfm(dist_source)

            dist_target = torch.zeros([bs_part, bs_part-1]).cuda()
            for pair1 in range(bs_part):
                tmpc = 0
                anchor_feat = torch.unsqueeze(model_pred_prior_wave[pair1].reshape(-1),0)
                for pair2 in range(bs_part):
                    if pair1 != pair2:
                        target_feat = torch.unsqueeze(model_pred_prior_wave[pair2].reshape(-1),0)
                        dist_target[pair1, tmpc] = sim(anchor_feat, target_feat)
                        tmpc += 1
            dist_target = sfm(dist_target)

            rel_wave_loss = kl_loss(torch.log(dist_source), dist_target)

            batch2 = next(iter(train_dataloader2))
            batch2_images = batch2["pixel_values"].to(device, non_blocking=True)
            data_instance, _ = torch.chunk(batch2_images, 2, dim=0)
            data_instance_wave = LH(data_instance)+HL(data_instance)+HH(data_instance)
            wave_loss = F.mse_loss(model_pred_image_wave.float(), data_instance_wave.float(), reduction="mean")

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the losses to the instance loss.
            loss = loss + prior_loss_weight * prior_loss + image_loss_weight * rel_loss \
                + hf_loss_weight * rel_wave_loss + hfmse_loss_weight * wave_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)