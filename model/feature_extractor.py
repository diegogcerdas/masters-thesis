import abc

import open_clip
import torch


# TODO: Add support for other embeddings + multiple embeddings
class FeatureExtractorType:
    CLIP = "clip"
    CLIP_MASKED = "clip_masked"  # TODO: Implement masking
    MERU = "meru"  # TODO: implement MERU
    VGG19 = "vgg19"  # TODO: Implement feature map extraction
    ALEXNET = "alexnet"


class FeatureExtractor(abc.ABC):
    @property
    def feature_size(self):
        """
        Returns the size of the feature vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self, data):
        """
        Extracts features from the given data and returns them as a feature vector.
        """
        raise NotImplementedError


class CLIPExtractor(FeatureExtractor):
    def __init__(self, model_name: str, pretrained: str):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.clip.requires_grad_(False)

    def feature_size(self):
        return self.clip.visual.output_dim

    def extract_features(self, data: torch.Tensor):
        with torch.no_grad():
            x = self.clip.encode_image(data)
            # TODO: normalize
        return x


def create_feature_extractor(type: FeatureExtractorType) -> FeatureExtractor:
    if type == FeatureExtractorType.CLIP:
        return CLIPExtractor(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
        )
