import abc
import os

import numpy as np
import open_clip
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset.natural_scenes import NaturalScenesDataset
from utils.custom_transforms import RandomSpatialOffset

from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader

import torch.nn as nn


# TODO: Add support for other embeddings
class FeatureExtractorType:
    CLIP = "clip"
    MERU = "meru"  # TODO: implement MERU
    ALEXNET_1 = "alexnet_1"
    ALEXNET_2 = "alexnet_2"
    ALEXNET_3 = "alexnet_3"
    ALEXNET_4 = "alexnet_4"
    ALEXNET_5 = "alexnet_5"
    ALEXNET_6 = "alexnet_6"
    DINOV2 = "dinov2"  # TODO: implement DINOv2
    SDVAE = "sdvae"  # TODO: implement Stable Diffusion VAE


class FeatureExtractor(nn.Module):
    def __init__(self, name: str, model_name: str, source: str, model_parameters: dict, module_name: str, mean: float, std: float, device: str = None):
        super().__init__()
        self.device = device
        self.name = name
        self.module_name = module_name
        self.mean = mean
        self.std = std

        self.extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=str(device),
            pretrained=True,
            model_parameters=model_parameters,
        )
        self.extractor.model.requires_grad_(False)

        self.feature_size = self.extractor.extract_features(
            batches=[torch.randn(1, 3, 425, 425).to(device)],
            module_name='features',
            flatten_acts=True,
            output_type="ndarray",
        ).shape[1]

        self.augmentation = transforms.Compose(
            [
                transforms.Lambda(lambda x: x * np.random.uniform(0.95, 1.05)),
                RandomSpatialOffset(offset=4),
            ]
        )

    def forward(self, x: torch.Tensor, mode: str = "val"):
        if mode == "train":
            x = self.augmentation(x)
        x = self.extractor.extract_features(
            batches=[x],
            module_name=self.module_name,
            flatten_acts=True,
            output_type="ndarray",
        )
        x = ((x - self.mean) / self.std).float()
        return x

    def extract_for_dataset(
        self, filename: str, dataset: NaturalScenesDataset, batch_size: int = 8
    ):
        assert not dataset.return_coco_id
        folder = os.path.dirname(filename)
        if not os.path.exists(filename):
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, backend='pt')
            features = self.extractor.extract_features(
                batches=dataloader,
                module_name=self.module_name,
                flatten_acts=True,
                output_type="ndarray",
            )
            os.makedirs(folder, exist_ok=True)
            np.save(filename, features)
        else:
            features = np.load(filename)
        return features


class CLIPExtractor(nn.Module):
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        super().__init__()
        self.device = device
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.clip.requires_grad_(False)
        self.feature_size = self.clip.visual.output_dim
        self.name = "clip"

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
        self.mean = -0.0057645994
        self.std = 0.5654832
        self.to(device)

    def forward(self, data: torch.Tensor, mode: str = "val"):
        if mode == "train":
            x = self.train_transform(data)
        else:
            x = self.test_transform(data)
        x = self.clip.encode_image(x)
        x = (x - self.mean) / self.std
        return x.float()
    
    def extract_for_dataset(
        self, filename: str, dataset: NaturalScenesDataset, batch_size: int = 8
    ):
        folder = os.path.dirname(filename)
        if not os.path.exists(filename):
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
                x = batch[0]
                x = x.to(self.device)
                bs = x.shape[0]
                x = self(x).reshape((bs, -1)).detach().cpu().numpy()
                features.append(x)
            features = np.concatenate(features, axis=0)
            os.makedirs(folder, exist_ok=True)
            np.save(filename, features)
        else:
            features = np.load(filename)
        return features


def create_feature_extractor(
    type: FeatureExtractorType, device: str = None
):
    if type == FeatureExtractorType.CLIP:
        feature_extractor = CLIPExtractor(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_1:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="features.2",
            mean=None,
            std=None,
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_2:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="features.5",
            mean=None,
            std=None,
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_3:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="features.7",
            mean=None,
            std=None,
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_4:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="features.9",
            mean=None,
            std=None,
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_5:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="avgpool",
            mean=None,
            std=None,
            device=device,
        )
    elif type == FeatureExtractorType.ALEXNET_6:
        feature_extractor = FeatureExtractor(
            name=type,
            model_name="alexnet",
            source="torchvision",
            model_parameters={'weights': 'DEFAULT'},
            module_name="classifier",
            mean=None,
            std=None,
            device=device,
        )
    return feature_extractor
