from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning import seed_everything
from peft import LoraConfig
import torch, os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from utils.img_utils import save_images
from tqdm import tqdm


def run(
    args_pretrained_model_name_or_path: str,
    args_data_dir: str,
    args_instance_prompt: str,
    args_num_timesteps: int,
    args_lora_rank: int,
    args_train_text_encoder: bool,
    args_validation_prompt: str,
    args_validation_epochs: int,
    args_num_val_images: int,
    args_save_folder: str,
    args_resolution: int,
    args_num_epochs: int,
    args_learning_rate: float,
    args_seed: int,
    args_device: str,
):
    seed_everything(args_seed)
    
    pipe = StableDiffusionPipeline.from_pretrained(args_pretrained_model_name_or_path)
    pipe = pipe.to(args_device)

    # freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # disable safety checker
    pipe.safety_checker = None   

    # switch to DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args_num_timesteps)

    # Now we will add new LoRA weights to the attention layers
    unet_lora_config_1 = LoraConfig(
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
    pipe.unet.add_adapter(unet_lora_config_1, adapter_name="lora_1")

    unet_lora_config_2 = LoraConfig(
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
    pipe.unet.add_adapter(unet_lora_config_2, adapter_name="lora_2")

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args_train_text_encoder:
        text_lora_config_1 = LoraConfig(
            r=args_lora_rank,
            lora_alpha=args_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipe.text_encoder.add_adapter(text_lora_config_1, adapter_name="lora_1")

        text_lora_config_2 = LoraConfig(
            r=args_lora_rank,
            lora_alpha=args_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipe.text_encoder.add_adapter(text_lora_config_2, adapter_name="lora_2")

    # Initialize the optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
    if args_train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, pipe.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args_learning_rate)

    # Prepare dataset
    filenames = [int(f.replace(".png", "")) for f in os.listdir(args_data_dir) if f.endswith(".png")]
    filenames = [os.path.join(args_data_dir, f"{n}.png") for n in sorted(filenames)]
    bins = np.linspace(0, 1, 11)
    alphas = bins[(np.digitize(np.linspace(0, 1.1, len(filenames)), bins) - 1)]
    image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    args_resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    for epoch in tqdm(range(args_num_epochs)):

        #################### VALIDATION ####################

        if epoch % args_validation_epochs == 0:

            pipe.unet.eval()
            if args_train_text_encoder:
                pipe.text_encoder.eval()
            
            for alpha in bins:

                pipe.set_adapters(["lora_1", "lora_2"], adapter_weights=[1 - alpha, alpha])

                pipeline_args = {
                    "prompt": args_validation_prompt,
                    "num_inference_steps": args_num_timesteps,
                }

                images = []
                generator = torch.Generator(device=args_device).manual_seed(args_seed)
                for _ in range(args_num_val_images):
                    with torch.cuda.amp.autocast():
                        image = pipe(**pipeline_args, generator=generator).images[0]
                        images.append(image)
            
                save_folder = os.path.join(args_save_folder, f'Epoch {epoch}', f'Alpha {alpha:.1f}')
                os.makedirs(save_folder, exist_ok=True)
                save_images(images, save_folder)


        #################### TRAINING ####################

        pipe.unet.train()
        if args_train_text_encoder:
            pipe.text_encoder.train()

        order = np.random.permutation(len(filenames))
        data_f = np.array(filenames)[order]
        data_a = np.array(alphas)[order]

        for file, alpha in zip(data_f, data_a):
            
            # Adjust adapter weights
            pipe.set_adapters(["lora_1", "lora_2"], adapter_weights=[1 - alpha, alpha])
            
            # Prepare model input
            img = image_transforms(Image.open(file)).unsqueeze(0).to(args_device)
            model_input = pipe.vae.encode(img).latent_dist.sample()
            model_input = model_input * pipe.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz, channels, _, _ = model_input.shape

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
            prompt_ids = pipe.tokenizer(
                args_instance_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipe.tokenizer.model_max_length,
            ).input_ids.to(args_device)       
            prompt_embeds = pipe.text_encoder(prompt_ids)[0]

            # Predict the noise residual
            if pipe.unet.config.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
            model_pred = pipe.unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {pipe.scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    