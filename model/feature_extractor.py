import abc

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms

from utils.custom_transforms import RandomSpatialOffset
from torch.utils import data
from tqdm import tqdm


# TODO: Add support for other embeddings
class FeatureExtractorType:
    CLIP = "clip"
    MERU = "meru"  # TODO: implement MERU
    ALEXNET = "alexnet"  # TODO: implement feature map extraction
    DINOV2 = "dinov2"  # TODO: implement DINOv2
    SDVAE = "sdvae"  # TODO: implement Stable Diffusion VAE


class FeatureExtractor(abc.ABC, torch.nn.Module):
    def __init__(self, device: str = None):
        super().__init__()
        self.device = device
        self.feature_size = ...
        self.name = ...

    def extract_for_dataset(self, dataset: data.Dataset, batch_size: int):
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
        embeddings = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            x = batch[0]
            x = x.to(self.device)
            bs = x.shape[0]
            x = self(x).reshape((bs, -1)).detach().cpu().numpy()
            embeddings.append(x)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


class CLIPExtractor(FeatureExtractor):
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        super().__init__(device=device)
        self.name = 'clip'
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.clip.requires_grad_(False)
        self.feature_size = self.clip.visual.output_dim

        self.train_transform = transforms.Compose(
            [
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
                transforms.Resize(224, antialias=True),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward(self, data: torch.Tensor, mode: str = "val"):
        if mode == "train":
            x = self.train_transform(data)
        else:
            x = self.test_transform(data)
        x = self.clip.encode_image(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


def create_feature_extractor(type: FeatureExtractorType, device: str = None) -> FeatureExtractor:
    if type == FeatureExtractorType.CLIP:
        return CLIPExtractor(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
