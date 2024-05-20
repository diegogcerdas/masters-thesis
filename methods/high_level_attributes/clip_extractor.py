import numpy as np
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.nsd.nsd import NaturalScenesDataset


class CLIPExtractorType:
    CLIP_1_5 = "clip_1_5"
    CLIP_2_0 = "clip_2_0"


class CLIPExtractor(nn.Module):
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

    def extract_for_dataset(self, dataset: NaturalScenesDataset):
        features = []
        for i in tqdm(range(len(dataset))):
            img = dataset[i][0]
            x = self(img).detach().cpu().numpy()
            features.append(x)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return features

    def forward(self, data: torch.Tensor, mode: str = "val"):
        if mode == "val":
            with torch.no_grad():
                x = self.transform(data).to(self.device).unsqueeze(0)
                x = self.clip.encode_image(x).reshape(x.shape[0], -1).float()
        elif mode == "train":
            x = self.transform(data).to(self.device).unsqueeze(0)
            x = self.clip.encode_image(x).reshape(x.shape[0], -1).float()
        x = (x - self.mean) / self.std
        return x


def create_clip_extractor(
    type: CLIPExtractorType, device: str = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
) -> CLIPExtractor:
    if type == CLIPExtractorType.CLIP_1_5:
        feature_extractor = CLIPExtractor(
            model_name="ViT-L-14",
            pretrained="openai",
            name="clip_1_5",
            device=device,
        )
    elif type == CLIPExtractorType.CLIP_2_0:
        feature_extractor = CLIPExtractor(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            name="clip_2_0",
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
