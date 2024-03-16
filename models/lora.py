import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin

from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

from models.stable_diffusion import StableDiffusion
from utils.img_utils import save_images


class LoRA(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        rank: int,
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
        super(LoRA, self).__init__()
        self.save_hyperparameters()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.my_device = device
        self.rank = rank
        self.learning_rate = learning_rate
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

        # initialize UNet LoRA
        unet_lora_attn_procs = {}
        for name, attn_processor in self.ldm.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.ldm.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.ldm.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.ldm.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.ldm.unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")
            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )
            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.rank
            )
        self.ldm.unet.set_attn_processor(unet_lora_attn_procs)
        self.unet_lora_layers = AttnProcsLayers(self.ldm.unet.attn_processors)
        self.ldm.unet.train()

    def configure_optimizers(self):
        params_to_optimize = (self.unet_lora_layers.parameters())
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
        )
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
            0,
            self.ldm.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=model_input.device,
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.ldm.noise_scheduler.add_noise(
            model_input, noise, timesteps
        )

        # Get the text embedding for conditioning
        input_ids = torch.cat(
            [self.ldm.tokenize_prompt(self.instance_prompt).input_ids for _ in batch],
            dim=0,
        )
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
            target = self.ldm.noise_scheduler.get_velocity(
                model_input, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.ldm.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)
    
    def sample_and_save(self, save_folder):
        pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            unet=self.ldm.unet,
            text_encoder=self.ldm.text_encoder,
        )
        pipeline_args = {
            "prompt": self.validation_prompt,
            "num_inference_steps": self.inference_steps,
        }

        images = self.ldm.sample(
            self.num_validation_images,
            pipeline,
            pipeline_args,
            self.my_device,
            self.seed,
        )

        os.makedirs(save_folder, exist_ok=True)
        save_images(images, save_folder)

        LoraLoaderMixin.save_lora_weights(
            save_directory=save_folder,
            unet_lora_layers=self.unet_lora_layers,
            text_encoder_lora_layers=None,
            safe_serialization=True,
        )
    
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.max_train_epochs - 1:
            save_folder = os.path.join(self.output_dir, "final")
            self.sample_and_save(save_folder)
        return super().on_train_epoch_end()
    
    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.validation_epochs == 0:
            save_folder = os.path.join(self.output_dir, f"epoch_{self.current_epoch}")
            self.sample_and_save(save_folder)
        return super().on_train_epoch_start()
