import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from datasets.nsd import NaturalScenesDataset


class FeatureExtractorType:
    CLIP = "clip"


class FeatureExtractor(nn.Module):
    feature_size = None
    name = None

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


class CLIPExtractor(FeatureExtractor):
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        super().__init__()
        self.device = device
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.feature_size = self.clip.visual.output_dim
        self.name = "clip"
        self.transform = transforms.Compose(
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

    def forward(self, data: Image.Image):
        with torch.no_grad():
            x = self.transform(data).to(self.device)
            x = self.clip.encode_image(x)
            x = (x - self.mean) / self.std
        return x.float()


def create_feature_extractor(
    type: FeatureExtractorType, device: str
) -> FeatureExtractor:
    if type == FeatureExtractorType.CLIP:
        feature_extractor = CLIPExtractor(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
