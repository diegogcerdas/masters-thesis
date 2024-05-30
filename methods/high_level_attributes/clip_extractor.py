import numpy as np
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from datasets.nsd.nsd import NaturalScenesDataset


class CLIPExtractorType:
    CLIP_1_5 = "clip_1_5"
    CLIP_2_0 = "clip_2_0"


class CLIPExtractor(nn.Module):
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        super().__init__()
        self.device = device
        self.clip, _, self.transform = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.transform.transforms.pop(-3)  # remove rgb transform
        self.transform.transforms.pop(-2)  # remove totensor transform
        self.to(device)

    def extract_for_dataset(self, dataset: NaturalScenesDataset):
        assert dataset.partition == "all"
        assert dataset.transform is None
        features = []
        for i in tqdm(range(len(dataset))):
            img = transforms.ToTensor(dataset[i][0]).float()
            x = self(img).detach().cpu().numpy()
            features.append(x)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return features

    @torch.no_grad()
    def forward(self, data: torch.Tensor):
        x = self.transform(data).to(self.device).unsqueeze(0)
        x = self.clip.encode_image(x).reshape(x.shape[0], -1).float()
        return x


def create_clip_extractor(
    type: CLIPExtractorType, device: str = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
) -> CLIPExtractor:
    if type == CLIPExtractorType.CLIP_1_5:
        feature_extractor = CLIPExtractor(
            model_name="ViT-L-14",
            pretrained="openai",
            device=device,
        )
    elif type == CLIPExtractorType.CLIP_2_0:
        feature_extractor = CLIPExtractor(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=device,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {type}")
    return feature_extractor
