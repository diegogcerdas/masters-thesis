import os
from typing import Literal, Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, SchedulerMixin
from safetensors.torch import save_file
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.image_processor import VaeImageProcessor


#################################################################################
# LoRA Network
#################################################################################

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict",  # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
]


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "full",
    ) -> None:
        super().__init__()

        self.lora_scale = 1
        self.multiplier = 1
        self.rank = rank
        self.alpha = alpha
        self.prefix = "lora_unet"
        self.target_replace_modules = ["Attention"]

        self.unet_loras = self.create_modules(
            unet,
            train_method=train_method,
        )

        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        root_module: nn.Module,
        train_method: TRAINING_METHODS,
    ) -> list:
        loras = []
        names = []
        for name, module in root_module.named_modules():
            if (
                train_method == "noxattn"
                or train_method == "noxattn-hspace"
                or train_method == "noxattn-hspace-last"
            ):
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":
                if "attn2" not in name:
                    continue
            elif train_method == "full":
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in self.target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in [
                        "Linear",
                        "Conv2d",
                        "LoRACompatibleLinear",
                        "LoRACompatibleConv",
                    ]:
                        if train_method == "xattn-strict":
                            if "out" in child_name:
                                continue
                        if train_method == "noxattn-hspace":
                            if "mid_block" not in name:
                                continue
                        if train_method == "noxattn-hspace-last":
                            if (
                                "mid_block" not in name
                                or ".1" not in name
                                or "conv2" not in child_name
                            ):
                                continue
                        lora_name = self.prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora = LoRAModule(
                            lora_name,
                            child_module,
                            self.multiplier,
                            self.rank,
                            self.alpha,
                        )
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0


#################################################################################
# Tokenization and encoding
#################################################################################


def text_tokenize(
    tokenizer: AutoTokenizer,
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: AutoTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)
    return text_embeddings


#################################################################################
# Train utils
#################################################################################


@torch.no_grad()
def get_noisy_image(
    img,
    vae,
    generator,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.preprocess(img).to(vae.device)
    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents
    noise = torch.randn(init_latents.shape, generator=generator, device=vae.device)
    timestep = scheduler.timesteps[total_timesteps:total_timesteps+1]
    init_latents = scheduler.add_noise(init_latents, noise, timestep)
    return init_latents, noise

def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    return guided_target

def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int = 1,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)