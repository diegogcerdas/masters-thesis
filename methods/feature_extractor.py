import numpy as np
import open_clip
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from torchvision import transforms

from datasets.nsd import NaturalScenesDataset


class FeatureExtractorType:
    CLIP_1_5 = "clip_1_5"
    CLIP_2_0 = "clip_2_0"
    VAE_1_5 = "vae_1_5"
    VAE_2_0 = "vae_2_0"


class FeatureExtractor(nn.Module):
    feature_size = None
    name = None

    def extract_for_dataset(self, dataset: NaturalScenesDataset):
        features = []
        for i in tqdm(range(len(dataset))):
            img, _, _ = dataset[i]
            x = self(img).detach().cpu().numpy()
            features.append(x)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return features


class CLIPExtractor(FeatureExtractor):
    def __init__(self, model_name: str, pretrained: str, name: str, device: str = None):
        super().__init__()
        self.device = device
        self.clip, _, self.transform = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.transform.transforms.pop(-3)  # remove rgb transform
        self.transform.transforms.pop(-2)  # remove totensor transform
        self.feature_size = self.clip.visual.output_dim
        self.name = name
        self.mean = 0
        self.std = 0.65
        self.to(device)

    def forward(self, data: torch.Tensor, mode: str = "val"):
        if mode == "val":
            with torch.no_grad():
                x = self.transform(data).to(self.device)
                x = self.clip.encode_image(x).reshape(x.shape[0], -1).float()
        elif mode == "train":
            x = self.transform(data).to(self.device)
            x = self.clip.encode_image(x).reshape(x.shape[0], -1).float()
        x = (x - self.mean) / self.std
        return x


class VAEExtractor(FeatureExtractor):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        resolution: int,
        name: str,
        mean: float,
        std: float,
        device: str = None,
    ):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.feature_size = (
            self.vae.config.latent_channels * (resolution // vae_scale_factor) ** 2
        )
        self.name = name
        self.mean = mean
        self.std = std
        self.to(device)

    def forward(self, data: torch.Tensor, mode: str = "val"):
        if mode == "val":
            with torch.no_grad():
                image = transforms.Resize((self.resolution, self.resolution))(data)
                image = self.image_processor.preprocess(image).to(self.device)
                latents = self.vae.encode(image).latent_dist.sample(None)
                latents = self.vae.config.scaling_factor * latents
                latents = latents.reshape(latents.shape[0], -1)
        elif mode == "train":
            image = transforms.Resize((self.resolution, self.resolution))(data)
            image = self.image_processor.preprocess(image).to(self.device)
            latents = self.vae.encode(image).latent_dist.sample(None)
            latents = self.vae.config.scaling_factor * latents
            latents = latents.reshape(latents.shape[0], -1)
        latents = (latents - self.mean) / self.std
        return latents


def create_feature_extractor(
    type: FeatureExtractorType, device: str
) -> FeatureExtractor:
    if type == FeatureExtractorType.CLIP_1_5:
        feature_extractor = CLIPExtractor(
            model_name="ViT-L-14",
            pretrained="openai",
            name="clip_1_5",
            device=device,
        )
    elif type == FeatureExtractorType.CLIP_2_0:
        feature_extractor = CLIPExtractor(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            name="clip_2_0",
            device=device,
        )
    elif type == FeatureExtractorType.VAE_1_5:
        feature_extractor = VAEExtractor(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            resolution=512,
            name="vae_1_5",
            mean=0.05,
            std=0.85,
            device=device,
        )
    elif type == FeatureExtractorType.VAE_2_0:
        feature_extractor = VAEExtractor(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
            resolution=768,
            name="vae_2_0",
            mean=0.1,
            std=0.95,
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
