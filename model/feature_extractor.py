import numpy as np
import open_clip
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset.natural_scenes import NaturalScenesDataset
from utils.custom_transforms import RandomSpatialOffset

from diffusers import AutoencoderKL

from PIL import Image
from diffusers.image_processor import VaeImageProcessor


class FeatureExtractorType:
    CLIP = "clip"
    VAE = "vae"


class CLIPExtractor(nn.Module):
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        super().__init__()
        self.device = device
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.feature_size = self.clip.visual.output_dim
        self.name = "clip"

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Lambda(lambda x: x * np.random.uniform(0.95, 1.05)),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
                RandomSpatialOffset(offset=4),
                transforms.Lambda(
                    lambda x: x + (torch.randn(x.shape) * 0.05**2).to(x.device)
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.mean = 0
        self.std = 0.5
        self.to(device)

    def forward(self, data: Image.Image, mode: str = "val"):
        if mode == "train":
            x = self.train_transform(data)
        else:
            x = self.test_transform(data)
        x = self.clip.encode_image(x)
        x = (x - self.mean) / self.std
        return x.float()

    def extract_for_dataset(self, dataset: NaturalScenesDataset, batch_size: int = 8):
        assert not dataset.return_coco_id
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
        features = []
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Extracting features..."
        ):
            x = batch
            x = x.to(self.device)
            bs = x.shape[0]
            x = self(x).reshape((bs, -1)).detach().cpu().numpy()
            features.append(x)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return features
    

class VAEExtractor(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str, resolution: int, device: str = None):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device)
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.feature_size = self.vae.config.latent_channels * (resolution // vae_scale_factor)**2
        self.name = f'vae_{pretrained_model_name_or_path}'
        self.to(device)

    def forward(self, data: Image.Image, mode: str = "val"):
        r = self.resolution
        img = self.image_processor.preprocess(data, height=r, width=r)
        x = self.vae.encode(img, return_dict=False)[0].mode().reshape(-1).float()
        return x

    def extract_for_dataset(self, dataset: NaturalScenesDataset, batch_size: int = 8):
        assert not dataset.return_coco_id
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
        features = []
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Extracting features..."
        ):
            x = batch
            x = x.to(self.device)
            bs = x.shape[0]
            x = self(x).reshape((bs, -1)).detach().cpu().numpy()
            features.append(x)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return features


def create_feature_extractor(type: FeatureExtractorType, device: str):
    if type == FeatureExtractorType.CLIP:
        feature_extractor = CLIPExtractor(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
    elif type == FeatureExtractorType.VAE:
        feature_extractor = VAEExtractor(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            resolution=512,
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
