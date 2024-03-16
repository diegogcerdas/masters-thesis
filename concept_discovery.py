import itertools
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from concept_discovery_utils import freeze_params, ComposableDataset


def run(
    args_pretrained_model_name_or_path: str,
    args_seed: int,
    args_train_data_dir: str,
    args_resolution: int,
    args_repeats: int,
    args_add_weight_per_score: bool,
    args_init_weight: float,
    args_learning_rate: float,
    args_train_batch_size: int,
    args_device: str,
    args_num_train_epochs: int,
    args_validation_epochs: int,
    args_output_dir: str,
    args_num_validation_images: int,
):
    placeholder_tokens = ['<t1>','<t2>','<t3>','<t4>','<t5>']
    
    pl.seed_everything(args_seed)

    tokenizer = CLIPTokenizer.from_pretrained(args_pretrained_model_name_or_path, subfolder="tokenizer")

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    if num_added_tokens != 0 and num_added_tokens != len(placeholder_tokens):
        raise ValueError(
            f"The tokenizer already contains at least one of the tokens in {placeholder_tokens}. "
            f"Please pass a different placeholder_token` that is not already in the tokenizer."
        )

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="text_encoder").to(args_device)
    vae = AutoencoderKL.from_pretrained(args_pretrained_model_name_or_path, subfolder="vae").to(args_device)
    unet = UNet2DConditionModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="unet").to(args_device)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    train_dataset = ComposableDataset(
        data_root=args_train_data_dir,
        tokenizer=tokenizer,
        size=args_resolution,
        repeats=args_repeats,
        placeholder_tokens=placeholder_tokens,
    )

    if args_add_weight_per_score:
        # Add a learnable weight for each token
        num_tokens = len(placeholder_token_ids)
        # create weight matrix NxMx1x1x1 where D is the number of images and M is the number of classes
        concept_weights = torch.tensor([args_init_weight] * num_tokens).reshape(1, -1, 1, 1, 1).float()
        concept_weights = concept_weights.repeat(train_dataset.num_images, 1, 1, 1, 1)
        concept_weights = torch.nn.Parameter(concept_weights, requires_grad=True)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(
            text_encoder.get_input_embeddings().parameters(),
            [concept_weights] if args_add_weight_per_score else []
        ),
        lr=args_learning_rate,
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args_train_batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler.from_pretrained(args_pretrained_model_name_or_path, subfolder="scheduler")

    # keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    generator = torch.Generator(device=args_device).manual_seed(args_seed)

    for epoch in tqdm(range(args_num_train_epochs)):

        text_encoder.train()

        for batch in train_dataloader:

            pixel_value, input_ids, weight_id = batch["pixel_values"], batch["input_ids"], batch["gt_weight_id"]
            pixel_value = pixel_value.to(args_device)
            input_ids = input_ids.to(args_device)
            input_ids_list = [y.squeeze(dim=1) for y in input_ids.chunk(chunks=input_ids.shape[1], dim=1)]

            # latents
            latents = vae.encode(pixel_value).latent_dist.sample().detach()
            latents = latents * 0.18215
            bsz = latents.shape[0]
            noise = torch.randn(latents.shape, generator=generator, device=args_device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            weights = concept_weights[weight_id]

            cond_scores = []
            for input_ids in input_ids_list:
                encoder_hidden_state = text_encoder(input_ids)[0]
                cond_scores.append(unet(noisy_latents, timesteps, encoder_hidden_state).sample)
            cond_scores = torch.stack(cond_scores, dim=1)
            uncond_text_ids = tokenizer(
                "",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(latents.device)
            uncond_encoder_hidden_states = text_encoder(uncond_text_ids)[0].repeat(bsz, 1, 1)
            uncond_score = unet(noisy_latents, timesteps, uncond_encoder_hidden_states).sample

            # compute initial compositional score
            composed_score = uncond_score + torch.sum(weights.to(latents.device) * (cond_scores - uncond_score[:, None]), dim=1)
            loss = F.mse_loss(noise, composed_score.float(), reduction="mean")

            loss.backward()
            
            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
            index_no_updates[placeholder_token_ids] = False
            grads = text_encoder.get_input_embeddings().weight.grad
            grads.data[index_no_updates, :] = grads.data[index_no_updates, :].fill_(0)

            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

            optimizer.step()
            optimizer.zero_grad()

        if epoch % args_validation_epochs == 0:
            
            folder = os.path.join(args_output_dir, f'epoch_{epoch}')
            os.makedirs(folder, exist_ok=True)

            pipeline = DiffusionPipeline.from_pretrained(
                args_pretrained_model_name_or_path,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                vae=vae,
            )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(args_device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=args_device).manual_seed(args_seed)
            for i in range(args_num_validation_images):
                for prompt in placeholder_tokens:
                    image = pipeline(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images[0]
                    image.save(os.path.join(folder, f'{prompt}_{i}.png'))

            del pipeline
            torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    pipeline = StableDiffusionPipeline.from_pretrained(
        args_pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
    )
    pipeline.save_pretrained(args_output_dir)