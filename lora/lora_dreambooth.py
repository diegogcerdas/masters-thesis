import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pytorch_lightning as pl
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)
from transformers import CLIPTextModel
from lora.lora_utils import DreamBoothDataset, collate_fn, encode_prompt, log_validation
from torch.utils import data

def run(
    args_pretrained_model_name_or_path: str,  # Path to pretrained model or model identifier from huggingface.co/models.
    args_instance_data_dir: str,  # A folder containing the training data of instance images.
    args_instance_prompt: str,  # The prompt with identifier specifying the instance.
    args_validation_prompt: str,  # A prompt that is used during validation to verify that the model is learning.
    args_num_validation_images: int,  # Number of images that should be generated during validation with `validation_prompt`.
    args_validation_epochs: int,  # Run dreambooth validation every X epochs.
    args_output_dir: str,  # The output directory where the model predictions and checkpoints will be written.
    args_seed: int,  # A seed for reproducible training.
    args_train_text_encoder: bool,  # Whether to train the text encoder. If set, the text encoder should be float32 precision.
    args_train_batch_size: int,  # Batch size for training.
    args_max_train_epochs: int,  # Number of training steps.
    args_learning_rate: float,  # The learning rate for the optimizer.
    args_lr_scheduler: str,  # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].
    args_dataloader_num_workers: int,  # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    args_rank: int = 4,  # The dimension of the LoRA update matrices.
    args_adam_beta1: float = 0.9,  # The beta1 hyperparameter for the Adam optimizer.
    args_adam_beta2: float = 0.999,  # The beta2 hyperparameter for the Adam optimizer.
    args_adam_weight_decay: float = 1e-2,  # The weight decay hyperparameter for the Adam optimizer.
    args_adam_epsilon: float = 1e-8,  # The epsilon hyperparameter for the Adam optimizer.
    args_resolution: int = 512,  # The resolution of the images.
    args_max_grad_norm: float = 1.0,  # Maximum gradient norm for clipping.
    args_device: str = "cuda",  # The device to use for training.
):

    pl.seed_everything(args_seed)
    os.makedirs(args_output_dir, exist_ok=True)

    # Load scheduler, tokenizer, and models
    noise_scheduler = DDPMScheduler.from_pretrained(args_pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(args_pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args_pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args_pretrained_model_name_or_path, subfolder="unet")

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    weight_dtype = torch.float32
    unet.to(args_device, dtype=weight_dtype)
    vae.to(args_device, dtype=weight_dtype)
    text_encoder.to(args_device, dtype=weight_dtype)

    # Now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args_rank,
        lora_alpha=args_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args_train_text_encoder:
        text_lora_config = LoraConfig(
            r=args_rank,
            lora_alpha=args_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args_train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args_learning_rate,
        betas=(args_adam_beta1, args_adam_beta2),
        weight_decay=args_adam_weight_decay,
        eps=args_adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args_instance_data_dir,
        instance_prompt=args_instance_prompt,
        tokenizer=tokenizer,
        size=args_resolution,
    )

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args_train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args_dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    lr_scheduler = get_scheduler(
        args_lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args_max_train_epochs * num_update_steps_per_epoch,
    )

    for epoch in tqdm(range(args_max_train_epochs)):

        unet.train()
        if args_train_text_encoder:
            text_encoder.train()

        for batch in train_dataloader:

            # Prepare model input
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype).to(args_device)
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz, channels, _, _ = model_input.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
            ).to(args_device)

            if unet.config.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute the loss and backpropagate
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, args_max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if args_validation_prompt is not None and epoch % args_validation_epochs == 0:
            
            # create pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                args_pretrained_model_name_or_path,
                unet=unet,
                text_encoder=text_encoder,
                torch_dtype=weight_dtype,
            )
            pipeline_args = {"prompt": args_validation_prompt, "num_inference_steps": 100}

            log_validation(
                args_num_validation_images,
                args_validation_prompt,
                pipeline,
                pipeline_args,
                args_output_dir,
                args_device,
                args_seed,
                epoch,
            )

            del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    unet = unet.to(torch.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    if args_train_text_encoder:
        text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    else:
        text_encoder_state_dict = None

    LoraLoaderMixin.save_lora_weights(
        save_directory=args_output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_state_dict,
    )

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args_pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    # load attention processors
    pipeline.load_lora_weights(args_output_dir, weight_name="pytorch_lora_weights.safetensors")

    # run inference
    if args_validation_prompt and args_num_validation_images > 0:
        pipeline_args = {"prompt": args_validation_prompt, "num_inference_steps": 100}
        log_validation(
                args_num_validation_images,
                args_validation_prompt,
                pipeline,
                pipeline_args,
                args_output_dir,
                args_device,
                args_seed,
                
        )
        del pipeline
        torch.cuda.empty_cache()