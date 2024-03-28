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
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict


def run(
    args_pretrained_model_name_or_path: str,
    args_data_dir: str,
    args_instance_prompt: str,
    args_invert_alphas: bool,
    args_num_timesteps: int,
    args_lora_rank: int,
    args_train_text_encoder: bool,
    args_validation_prompt: str,
    args_validation_epochs: int,
    args_num_val_images: int,
    args_save_folder: str,
    args_resolution: int,
    args_num_epochs: int,
    args_num_inner_epochs: int,
    args_batch_size: int,
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
    pipe.unet.add_adapter(unet_lora_config, adapter_name="lora")

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args_train_text_encoder:
        text_lora_config = LoraConfig(
            r=args_lora_rank,
            lora_alpha=args_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipe.text_encoder.add_adapter(text_lora_config, adapter_name="lora")

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
    if args_invert_alphas:
        alphas = alphas[::-1]
    args_batch_size = min(args_batch_size, len(np.where(alphas == alphas[0])[0]))
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

                pipeline_args = {
                    "prompt": args_validation_prompt,
                    "num_inference_steps": args_num_timesteps,
                }

                cross_attention_kwargs = {'scale': alpha}

                images = []
                generator = torch.Generator(device=args_device).manual_seed(args_seed)
                for _ in range(args_num_val_images):
                    with torch.cuda.amp.autocast():
                        image = pipe(**pipeline_args, generator=generator, cross_attention_kwargs=cross_attention_kwargs).images[0]
                        images.append(image)
            
                save_folder = os.path.join(args_save_folder, f'Epoch {epoch}')
                save_folder_alpha = os.path.join(save_folder, f'Alpha {alpha:.1f}')
                os.makedirs(save_folder_alpha, exist_ok=True)
                save_images(images, save_folder_alpha)
        
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(pipe.unet)
            )

            if args_train_text_encoder:
                text_encoder_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipe.text_encoder)
                )
            else:
                text_encoder_state_dict = None

            os.makedirs(save_folder, exist_ok=True)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_folder,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )

        #################### TRAINING ####################

        pipe.unet.train()
        if args_train_text_encoder:
            pipe.text_encoder.train()

        perm_bins = np.random.permutation(bins)

        for inner_epoch in range(args_num_inner_epochs):

            chosen_alpha = perm_bins[inner_epoch % len(perm_bins)]
            chosen_idxs = np.random.choice(np.where(alphas == chosen_alpha)[0], size=args_batch_size, replace=False)
            imgs = [Image.open(f).convert("RGB") for f in np.array(filenames)[chosen_idxs]]
            imgs = torch.stack([image_transforms(img).to(args_device) for img in imgs])
            
            # Prepare model input
            model_input = pipe.vae.encode(imgs).latent_dist.sample()
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
            prompt_embeds, _ = pipe.encode_prompt(
                args_instance_prompt,
                device=args_device,
                num_images_per_prompt=bsz,
                do_classifier_free_guidance=False,
                lora_scale=chosen_alpha,
            )

            # Predict the noise residual
            if pipe.unet.config.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
            model_pred = pipe.unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                cross_attention_kwargs = {'scale': chosen_alpha},
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