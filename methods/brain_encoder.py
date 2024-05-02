from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
from sklearn.linear_model import LinearRegression
from methods.feature_extractor import create_feature_extractor
from torchvision import transforms

class Encoder:

    def __init__(self, feature_extractor, linear_model) -> None:
        self.feature_extractor = feature_extractor
        self.linear_model = linear_model

    def __call__(self, img):
        feats = self.feature_extractor(transforms.ToTensor()(img).unsqueeze(0)).detach().cpu().numpy()
        pred = self.linear_model.predict(feats)
        return pred

def get_encoder(data_root, subject, roi, hemisphere, feature_extractor_type, metric, seed, device):
    nsd = NaturalScenesDataset(
        root=data_root,
        subject=subject,
        partition="train",
        hemisphere=hemisphere,
        roi=roi,
    )
    dataset = NSDFeaturesDataset(
        nsd=nsd,
        feature_extractor_type=feature_extractor_type,
        predict_average=True,
        metric=metric,
        n_neighbors=0,
        seed=seed,
        device=device,
        keep_features=True,
    )
    linear_model = LinearRegression().fit(dataset.features, dataset.targets)
    feature_extractor = create_feature_extractor(feature_extractor_type, device)

    return Encoder(feature_extractor, linear_model)