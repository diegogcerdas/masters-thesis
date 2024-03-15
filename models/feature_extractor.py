import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from datasets.nsd import NaturalScenesDataset

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


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
        self.feature_size = self.clip.visual.output_dim
        self.name = name
        self.to(device)

    def forward(self, data: Image.Image):
        with torch.no_grad():
            x = self.transform(data).unsqueeze(0).to(self.device)
            x = self.clip.encode_image(x).reshape(1,-1).float()
        return x
    

class VAEExtractor(FeatureExtractor):
    def __init__(self, pretrained_model_name_or_path: str, resolution: int, name: str, device: str = None):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.feature_size = self.vae.config.latent_channels * (resolution // vae_scale_factor)**2
        self.name = name
        self.to(device)

    def forward(self, data: Image.Image):
        with torch.no_grad():
            image = data.resize((self.resolution, self.resolution))
            image = self.image_processor.preprocess(image).to(self.device)
            latents = self.vae.encode(image).latent_dist.sample(None)
            latents = self.vae.config.scaling_factor * latents
            latents = latents.reshape(1, -1)
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
            device=device,
        )
    elif type == FeatureExtractorType.VAE_2_0:
        feature_extractor = VAEExtractor(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
            resolution=768,
            name="vae_2_0",
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
