from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning import seed_everything
from peft import LoraConfig
import torch, os
import torch.nn.functional as F
from utils.img_utils import save_images
from tqdm import tqdm
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from datasets.lora_dataset import LoRADataset
import torch.utils.data as data
import uuid


def train_lora(
    pretrained_model_name_or_path: str,
    data_dir: str,
    instance_prompt: str,
    num_timesteps: int,
    lora_rank: int,
    omit_unet: bool,
    omit_text_encoder: bool,
    validation_prompt: str,
    validation_epochs: int,
    num_val_images: int,
    save_folder: str,
    resolution: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    seed: int,
    device: str,
    with_prior_preservation_loss: bool = False,
    prior_preservation_loss_weight: float = 0.25,
    num_class_images: int = 200,
    class_data_dir: str = None,
    class_prompt: str = None,
):
    assert not (omit_unet and omit_text_encoder), "At least one of the models must be trainable."
    assert not with_prior_preservation_loss or (class_data_dir is not None and class_prompt is not None and num_class_images > 0)
    assert (not with_prior_preservation_loss and class_data_dir is None) or with_prior_preservation_loss
    seed_everything(seed)
    
    # Load the pretrained model
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Disable safety checker
    pipe.safety_checker = None   

    # Switch to DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_timesteps)

    # Generate class images if prior preservation is enabled.
    if with_prior_preservation_loss:
        
        os.makedirs(class_data_dir, exist_ok=True)
        cur_class_images = len([f for f in os.listdir(class_data_dir) if f.endswith(".png")])

        if cur_class_images < num_class_images:

            num_new_images = num_class_images - cur_class_images
            pipeline_args = {
                "prompt": class_prompt,
                "num_inference_steps": num_timesteps,
            }
            images = []
            generator = torch.Generator(device=device).manual_seed(seed)
            for _ in tqdm(range(num_new_images), desc="Generating images"):
                with torch.cuda.amp.autocast():
                    image = pipe(**pipeline_args, generator=generator).images[0]
                    images.append(image)

            names = [f'{str(uuid.uuid4())}.png' for _ in images]
            save_images(images, class_data_dir, names)
            del images

    # Add new LoRA weights to UNet
    if not omit_unet:
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
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
        pipe.unet.add_adapter(unet_lora_config)

    # Add new LoRA weights to text encoder
    if not omit_text_encoder:
        text_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipe.text_encoder.add_adapter(text_lora_config)

    # Initialize the optimizer
    params_to_optimize = []
    if not omit_unet:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
    if not omit_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipe.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    # Prepare dataset
    dataset = LoRADataset(data_dir, resolution, class_data_dir)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        #################### VALIDATION ####################

        if epoch % validation_epochs == 0:
            
            if not omit_unet:
                pipe.unet.eval()
            if not omit_text_encoder:
                pipe.text_encoder.eval()

            # Run the pipeline
            pipeline_args = {
                "prompt": validation_prompt,
                "num_inference_steps": num_timesteps,
            }
            images = []
            generator = torch.Generator(device=device).manual_seed(seed)
            for _ in tqdm(range(num_val_images), desc="Generating images"):
                with torch.cuda.amp.autocast():
                    image = pipe(**pipeline_args, generator=generator).images[0]
                    images.append(image)

            # Save the images
            save_folder_epoch = os.path.join(save_folder, f'Epoch {epoch}')
            save_images(images, save_folder_epoch)

            # Save LoRA weights
            if not omit_unet:
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipe.unet)
                )
            else:
                unet_lora_state_dict = None

            if not omit_text_encoder:
                text_encoder_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipe.text_encoder)
                )
            else:
                text_encoder_state_dict = None

            LoraLoaderMixin.save_lora_weights(
                save_directory=save_folder_epoch,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )

        #################### TRAINING ####################

        if not omit_unet:
            pipe.unet.train()
        if not omit_text_encoder:
            pipe.text_encoder.train()

        for batch in dataloader:
            
            # Prepare model input
            if with_prior_preservation_loss:
                batch = torch.cat([batch[0], batch[1]], dim=0)
            batch = batch.to(device)
            model_input = pipe.vae.encode(batch).latent_dist.sample()
            model_input = model_input * pipe.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz, _, _, _ = model_input.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (bsz,),
                device=model_input.device,
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = pipe.scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            prompt_embeds, _ = pipe.encode_prompt(
                instance_prompt,
                device=device,
                num_images_per_prompt=bsz,
                do_classifier_free_guidance=False,
            )

            # Predict the noise residual
            model_pred = pipe.unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                return_dict=False,
            )[0]

            # Get the target for loss depending on the prediction type
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {pipe.scheduler.config.prediction_type}"
                )

            if with_prior_preservation_loss:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + prior_preservation_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    