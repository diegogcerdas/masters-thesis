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
from torchvision import transforms
import random
import torch.utils.checkpoint as checkpoint

def run(
    args_pretrained_model_name_or_path: str,
    args_instance_prompt: str,
    args_num_timesteps: int,
    args_guidance_scale: float,
    args_lora_rank: int,
    args_train_text_encoder: bool,
    args_learning_rate: float,
    args_batch_size: int,
    args_num_train_iterations: int,
    args_validation_iters: int,
    args_validation_prompt: str,
    args_validation_images: int,
    args_brain_encoder_ckpt: str,
    args_save_folder: str,
    args_seed: int,
    args_device: str,
):
    pl.seed_everything(args_seed)
    
    pipeline = StableDiffusionPipeline.from_pretrained(args_pretrained_model_name_or_path)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    # disable safety checker
    pipeline.safety_checker = None    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(args_num_timesteps)

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
    pipeline.unet.train()

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args_train_text_encoder:
        text_lora_config = LoraConfig(
            r=args_lora_rank,
            lora_alpha=args_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipeline.text_encoder.add_adapter(text_lora_config)    
        pipeline.text_encoder.train() 
    
    # Initialize the optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    if args_train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipeline.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args_learning_rate)
    
    # load brain encoder
    loss_fn = EncoderModule.load_from_checkpoint(args_brain_encoder_ckpt, map_location=args_device)

    # generate unconditional embeddings
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

    prompt_embeds = pipeline.text_encoder(
    pipeline.tokenizer(
        args_instance_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(args_device)
    )[0]   
    cond_prompt_embeds = prompt_embeds.repeat(args_batch_size, 1, 1)
            
    for iter_i in tqdm(range(args_num_train_iterations)):

        #################### VALIDATION ####################

        if iter_i % args_validation_iters == 0:
            
            pipeline_val = StableDiffusionPipeline.from_pretrained(
                args_pretrained_model_name_or_path,
                unet=pipeline.unet,
                text_encoder=pipeline.text_encoder,
            )
            pipeline_args = {
                "prompt": args_validation_prompt,
                "num_inference_steps": args_num_timesteps,
            }

            pipeline_val.scheduler = DDIMScheduler.from_config(pipeline_val.scheduler.config)
            pipeline_val = pipeline_val.to(args_device)
            pipeline_val.set_progress_bar_config(disable=True)

            # Run inference
            generator = torch.Generator(device=args_device).manual_seed(args_seed)
            images = []
            for _ in range(args_validation_images):
                with torch.cuda.amp.autocast():
                    image = pipeline_val(**pipeline_args, generator=generator).images[0]
                    images.append(image)

            batch = []
            for image in images:
                batch.append(transforms.ToTensor()(image).unsqueeze(0))
            batch = torch.cat(batch, dim=0).to(args_device)
            pred = loss_fn(batch, mode='val').mean()
            print(f'{iter_i}: Validation pred: {pred.item()}')

            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(pipeline.unet)
            )

            if args_train_text_encoder:
                text_encoder_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipeline.text_encoder)
                )
            else:
                text_encoder_state_dict = None

            save_folder = os.path.join(args_save_folder, f'Iters {iter_i}')
            os.makedirs(save_folder, exist_ok=True)
            save_images(images, save_folder)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_folder,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )          

        #################### TRAINING ####################
            
        latent = torch.randn((args_batch_size, 4, 64, 64), device=args_device)                           

        for i, t in enumerate(pipeline.scheduler.timesteps):

            t = torch.tensor([t], device=args_device).repeat(args_batch_size)

            uncond_prompt_embeds = uncond_prompt_embeds.detach().clone()
            cond_prompt_embeds = cond_prompt_embeds.detach().clone()
            noise_pred_uncond = checkpoint.checkpoint(pipeline.unet, latent, t, uncond_prompt_embeds, use_reentrant=False).sample
            noise_pred_cond = checkpoint.checkpoint(pipeline.unet, latent, t, cond_prompt_embeds, use_reentrant=False).sample

            timestep = random.randint(0, args_num_timesteps)
            if i < timestep:
                noise_pred_uncond = noise_pred_uncond.detach()
                noise_pred_cond = noise_pred_cond.detach()

            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + args_guidance_scale * grad   
            
            latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                
        ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
        
        loss = -loss_fn(ims, mode='train', no_grad=False).mean()
        
        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()    