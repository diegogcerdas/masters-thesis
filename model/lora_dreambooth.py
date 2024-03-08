import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig
from model.stable_diffusion import StableDiffusion
from diffusers import DiffusionPipeline
import os
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)
from peft.utils import get_peft_model_state_dict
from diffusers.loaders import LoraLoaderMixin
from utils.img_utils import save_images

class LoRADreamBooth(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        rank: int,
        train_text_encoder: bool,
        instance_prompt: str,
        validation_prompt: str,
        num_validation_images: int,
        validation_epochs: int,
        max_train_epochs: int,
        inference_steps: int,
        output_dir: str,
        seed: int,
        learning_rate: float,
        device: str,
    ):
        super(LoRADreamBooth, self).__init__()
        self.save_hyperparameters()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.my_device = device
        self.rank = rank
        self.learning_rate = learning_rate
        self.train_text_encoder = train_text_encoder
        self.instance_prompt = instance_prompt
        self.validation_prompt = validation_prompt
        self.num_validation_images = num_validation_images
        self.validation_epochs = validation_epochs
        self.max_train_epochs = max_train_epochs
        self.inference_steps = inference_steps
        self.output_dir = output_dir
        self.seed = seed
        self.ldm = StableDiffusion(pretrained_model_name_or_path, device=device)
        self.prepare_for_lora()

    def prepare_for_lora(self):
        # We only train the additional adapter LoRA layers
        self.ldm.vae.requires_grad_(False)
        self.ldm.text_encoder.requires_grad_(False)
        self.ldm.unet.requires_grad_(False)

        # Now we will add new LoRA weights to the attention layers
        unet_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        self.ldm.unet.add_adapter(unet_lora_config)

        # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
        if self.train_text_encoder:
            text_lora_config = LoraConfig(
                r=self.rank,
                lora_alpha=self.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.ldm.text_encoder.add_adapter(text_lora_config)

    def configure_optimizers(self):
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.ldm.unet.parameters()))
        if self.train_text_encoder:
            params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, self.ldm.text_encoder.parameters()))
        optimizer = torch.optim.AdamW( params_to_optimize, lr=self.learning_rate,)
        return optimizer

    def compute_loss(self, batch):
        # Prepare model input
        model_input = self.ldm.vae.encode(batch).latent_dist.sample()
        model_input = model_input * self.ldm.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, _, _ = model_input.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.ldm.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.ldm.noise_scheduler.add_noise(model_input, noise, timesteps)

        # Get the text embedding for conditioning
        input_ids = torch.cat([self.ldm.tokenize_prompt(self.instance_prompt).input_ids for _ in batch], dim=0)
        encoder_hidden_states = self.ldm.encode_prompt(
            input_ids,
        ).to(self.my_device)

        # Predict the noise residual
        if self.ldm.unet.config.in_channels == channels * 2:
            noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
        model_pred = self.ldm.unet(
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
        if self.ldm.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.ldm.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.ldm.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.ldm.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log_stat("train/loss", loss)

        return loss

    def log_stat(self, name, stat):
        self.log(
            name,
            stat,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)
    
    def on_train_epoch_end(self) -> None:
        if (self.current_epoch % self.validation_epochs == 0) or self.current_epoch == self.max_train_epochs - 1:

            pipeline = DiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=self.ldm.unet,
                text_encoder=self.ldm.text_encoder,
            )
            pipeline_args = {"prompt": self.validation_prompt, "num_inference_steps": self.inference_steps}

            images = self.ldm.sample(
                self.num_validation_images,
                pipeline,
                pipeline_args,
                self.my_device,
                self.seed,
            )

            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.ldm.unet))

            if self.train_text_encoder:
                text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.ldm.text_encoder))
            else:
                text_encoder_state_dict = None

            save_folder = os.path.join(self.output_dir, f'epoch_{self.current_epoch}')
            if self.current_epoch == self.max_train_epochs - 1:
                save_folder = os.path.join(self.output_dir, 'final')
            os.makedirs(save_folder, exist_ok=True)
            save_images(images, save_folder)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_folder,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )
        return super().on_train_epoch_end()
    
