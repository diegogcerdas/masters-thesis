import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from aesthetic_scorer import AestheticScorerDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/align_prop.py", "Training configuration.")
from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)

import pytorch_lightning as pl
from peft import LoraConfig


def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
    

def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn



def evaluate(latent,train_neg_prompt_embeds,prompts, pipeline, accelerator, inference_dtype, config, loss_fn):
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)       
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
    
    all_rgbs_t = []
    for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), total=len(pipeline.scheduler.timesteps)):
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = pipeline.unet(latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = pipeline.unet(latent, t, prompt_embeds).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
    if "hps" in config.reward_fn:
        loss, rewards = loss_fn(ims, prompts)
    else:    
        _, rewards = loss_fn(ims)
    return ims, rewards

    
    

def main(
    args_pretrained_model_name_or_path: str,
    args_num_timesteps: int,
    args_lora_rank: int,
    args_lora_alpha: int,
    args_train_text_encoder: bool,
    args_learning_rate: float,
    args_seed: int,
    args_device: str,
):
    pl.seed_everything(args_seed)
    
    pipeline = StableDiffusionPipeline.from_pretrained(args_pretrained_model_name_or_path).to(args_device)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    # disable safety checker
    pipeline.safety_checker = None    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(args_num_timesteps)

    # Now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args_lora_rank,
        lora_alpha=args_lora_alpha,
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
            lora_alpha=args_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipeline.text_encoder.add_adapter(text_lora_config)        
    
    # Initialize the optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    if args_train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipeline.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args_learning_rate)
    
    # load brain encoder
    loss_fn = 
       
    global_step = 0

    #################### TRAINING ####################        
    for epoch in list(range(first_epoch, config.num_epochs)):
        unet.train()
        info = defaultdict(list)
        info_vis = defaultdict(list)
        image_vis_list = []
        
        for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_local_main_process):
            latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)    

            if accelerator.is_main_process:

                logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")

            
            prompts, prompt_metadata = zip(
                *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
            )

            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)   

            pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
            
        
            with accelerator.accumulate(unet):
                with autocast():
                    with torch.enable_grad(): # important b/c don't have on by default in module                        

                        keep_input = True
                        timesteps = pipeline.scheduler.timesteps
                        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                            t = torch.tensor([t],
                                                dtype=inference_dtype,
                                                device=latent.device)
                            t = t.repeat(config.train.batch_size_per_gpu_available)
                            
                            if config.grad_checkpoint:
                                noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                            else:
                                noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                                noise_pred_cond = unet(latent, t, prompt_embeds).sample
                                                            
                            if config.truncated_backprop:
                                if config.truncated_backprop_rand:
                                    timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                    if i < timestep:
                                        noise_pred_uncond = noise_pred_uncond.detach()
                                        noise_pred_cond = noise_pred_cond.detach()
                                else:
                                    if i < config.trunc_backprop_timestep:
                                        noise_pred_uncond = noise_pred_uncond.detach()
                                        noise_pred_cond = noise_pred_cond.detach()

                            grad = (noise_pred_cond - noise_pred_uncond)
                            noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad                
                            latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                                
                        ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                        
                        if "hps" in config.reward_fn:
                            loss, rewards = loss_fn(ims, prompts)
                        else:
                            loss, rewards = loss_fn(ims)
                        
                        loss =  loss.sum()
                        loss = loss/config.train.batch_size_per_gpu_available
                        loss = loss * config.train.loss_coeff

                        rewards_mean = rewards.mean()
                        rewards_std = rewards.std()
                        
                        if len(info_vis["image"]) < config.max_vis_images:
                            info_vis["image"].append(ims.clone().detach())
                            info_vis["rewards_img"].append(rewards.clone().detach())
                            info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                        
                        info["loss"].append(loss)
                        info["rewards"].append(rewards_mean)
                        info["rewards_std"].append(rewards_std)
                        
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()                        

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                assert (
                    inner_iters + 1
                ) % config.train.gradient_accumulation_steps == 0
                # log training and evaluation 
                if config.visualize_eval and (global_step % config.vis_freq ==0):

                    all_eval_images = []
                    all_eval_rewards = []
                    if config.same_evaluation:
                        generator = torch.cuda.manual_seed(config.seed)
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                    else:
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)                                
                    with torch.no_grad():
                        for index in range(config.max_vis_images):
                            ims, rewards = evaluate(latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],train_neg_prompt_embeds, eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], pipeline, accelerator, inference_dtype,config, loss_fn)
                            all_eval_images.append(ims)
                            all_eval_rewards.append(rewards)
                    eval_rewards = torch.cat(all_eval_rewards)
                    eval_reward_mean = eval_rewards.mean()
                    eval_reward_std = eval_rewards.std()
                    eval_images = torch.cat(all_eval_images)
                    eval_image_vis = []
                    if accelerator.is_main_process:

                        name_val = wandb.run.name
                        log_dir = f"logs/{name_val}/eval_vis"
                        os.makedirs(log_dir, exist_ok=True)
                        for i, eval_image in enumerate(eval_images):
                            eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            prompt = eval_prompts[i]
                            pil.save(f"{log_dir}/{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                            pil = pil.resize((256, 256))
                            reward = eval_rewards[i]
                            eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
                        accelerator.log({"eval_images": eval_image_vis},step=global_step)
                
                logger.info("Logging")
                
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
                accelerator.log(info, step=global_step)

                if config.visualize_train:
                    ims = torch.cat(info_vis["image"])
                    rewards = torch.cat(info_vis["rewards_img"])
                    prompts = info_vis["prompts"]
                    images  = []
                    for i, image in enumerate(ims):
                        image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((256, 256))
                        prompt = prompts[i]
                        reward = rewards[i]
                        images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                    
                    accelerator.log(
                        {"images": images},
                        step=global_step,
                    )

                global_step += 1
                info = defaultdict(list)

        # make sure we did an optimization step at the end of the inner epoch
        assert accelerator.sync_gradients
        
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

if __name__ == "__main__":
    app.run(main)