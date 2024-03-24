import torch
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import pytorch_lightning as pl
from peft import LoraConfig
from models.brain_encoder import EncoderModule
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from diffusers.loaders import LoraLoaderMixin
from utils.img_utils import save_images
from ddpo_utils import ddim_step_with_logprob, pipeline_with_logprob, get_prompt_fn


def run(
    args_pretrained_model_name_or_path: str,
    args_lora_rank: int,
    args_train_text_encoder: bool,
    args_brain_encoder_ckpt: str,
    args_prompt_filename: str,
    args_guidance_scale: float,
    args_eta: float,
    args_num_timesteps: int,
    args_num_epochs: int,
    args_num_inner_epochs: int,
    args_validation_epochs: int,
    args_save_folder: str,
    args_adv_clip_max: float,
    args_clip_range: float,
    args_batch_size: int,
    args_learning_rate: float,
    args_seed: int,
    args_device: str,
):

    pl.seed_everything(args_seed)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(args_pretrained_model_name_or_path)

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    # disable safety checker
    pipeline.safety_checker = None

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move unet, vae and text_encoder to device
    pipeline.vae.to(args_device)
    pipeline.text_encoder.to(args_device)
    pipeline.unet.to(args_device)

    # Now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args_lora_rank,
        lora_alpha=args_lora_rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ],
    )
    pipeline.unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args_train_text_encoder:
        text_lora_config = LoraConfig(
            r=args_lora_rank,
            lora_alpha=args_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipeline.text_encoder.add_adapter(text_lora_config)    

    # Initialize the optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    if args_train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipeline.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args_learning_rate)

    # prepare prompt and reward fn
    prompt_fn = get_prompt_fn(args_prompt_filename)
    reward_fn = EncoderModule.load_from_checkpoint(args_brain_encoder_ckpt, map_location=args_device)

    # generate unconditional prompt embeddings
    prompt_embeds = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(args_device)
    )[0]
    uncond_prompt_embeds = prompt_embeds.repeat(args_batch_size, 1, 1)

    # Train!
    for epoch in range(args_num_epochs):
        
        #################### SAMPLING ####################
        
        pipeline.unet.eval()
        if args_train_text_encoder:
            pipeline.text_encoder.eval()

        # generate prompts
        prompts = prompt_fn(args_batch_size)

        # encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(args_device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

        # sample
        images, _, latents, log_probs = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=uncond_prompt_embeds,
            num_inference_steps=args_num_timesteps,
            guidance_scale=args_guidance_scale,
            eta=args_eta,
            output_type="pt",
        )

        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
        timesteps = pipeline.scheduler.timesteps.repeat(args_batch_size, 1)  # (batch_size, num_steps)

        # compute reward
        rewards = reward_fn(images, mode='val')

        samples = {
            "prompt_embeds": prompt_embeds,
            "timesteps": timesteps,
            "latents": latents[:, :-1],  # each entry is the latent before timestep t
            "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
            "log_probs": log_probs,
            "advantages": (rewards - rewards.mean()) / (rewards.std() + 1e-8),
        }

        # log rewards
        print(f'{epoch}: reward mean: {rewards.mean().item()}, std: {rewards.std().item()}')

        if epoch % args_validation_epochs == 0:

            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(pipeline.unet)
            )

            if args_train_text_encoder:
                text_encoder_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipeline.text_encoder)
                )
            else:
                text_encoder_state_dict = None

            images = None

            save_folder = os.path.join(args_save_folder, f'Epoch {epoch}')
            os.makedirs(save_folder, exist_ok=True)
            save_images(images, save_folder)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_folder,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )          

        #################### TRAINING ####################
            
        for inner_epoch in range(args_num_inner_epochs):
            
            # shuffle samples along batch dimension
            perm = torch.randperm(args_batch_size, device=args_device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack([
                torch.randperm(args_num_timesteps, device=args_device)
                for _ in range(args_batch_size)
            ])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(args_batch_size, device=args_device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, args_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            if args_train_text_encoder:
                pipeline.text_encoder.train()

            for i, sample in tqdm(enumerate(samples_batched), desc=f"Epoch {epoch}.{inner_epoch}: training"):
                
                embeds = torch.cat([uncond_prompt_embeds, sample["prompt_embeds"]])

                for j in tqdm(range(args_num_timesteps), desc="Timestep"):

                    noise_pred = pipeline.unet(
                        torch.cat([sample["latents"][:, j]] * 2),
                        torch.cat([sample["timesteps"][:, j]] * 2),
                        embeds,
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + args_guidance_scale
                        * (noise_pred_text - noise_pred_uncond)
                    )

                    # compute the log prob of next_latents given latents under the current model
                    _, log_prob = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_pred,
                        sample["timesteps"][:, j],
                        sample["latents"][:, j],
                        eta=args_eta,
                        prev_sample=sample["next_latents"][:, j],
                    )

                    # ppo logic
                    advantages = torch.clamp(
                        sample["advantages"],
                        -args_adv_clip_max,
                        args_adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - args_clip_range,
                        1.0 + args_clip_range,
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    # backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()