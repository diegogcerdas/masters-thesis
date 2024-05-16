import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torchvision import transforms

from datasets.nsd_clip import NSDFeaturesDataset
from models.feature_extractor import FeatureExtractor


def encode(
    dataset: NSDFeaturesDataset,
    with_rsa: bool = True,
) -> tuple[LinearRegression, np.ndarray]:
    if with_rsa:
        # Compute RSA before Linear Regression
        voxel_RDM = (
            dataset.targets.reshape(-1, 1) - dataset.targets.reshape(1, -1)
        ).abs()
        idx = torch.triu_indices(*voxel_RDM.shape, offset=1)
        voxel_RDM = voxel_RDM[idx[0], idx[1]].numpy()
        rep_RDM = dataset.D[idx[0], idx[1]]
        r = spearmanr(voxel_RDM, rep_RDM).statistic
        print(f"RSA before Linear Regression: {round(r, 4)}")
    # Linear Regression
    encoder = LinearRegression().fit(dataset.features, dataset.targets)
    y_pred = encoder.predict(dataset.features)
    metric = encoder.score(dataset.features, dataset.targets)
    print(f"R^2: {round(metric, 4)}")
    if with_rsa:
        # Compute RSA after Linear Regression
        voxel_RDM = np.abs(y_pred.reshape(-1, 1) - y_pred.reshape(1, -1))
        voxel_RDM = voxel_RDM[idx[0], idx[1]]
        r = spearmanr(voxel_RDM, rep_RDM).statistic
        print(f"RSA after Linear Regression: {round(r, 4)}")
    return encoder, y_pred


def predict_from_dir(
    encoder: LinearRegression,
    feature_extractor: FeatureExtractor,
    preds_dir: str,
) -> None:
    # Predict activations
    data = []
    for folder in os.listdir(preds_dir):
        # Skip non-folders
        if not os.path.isdir(os.path.join(preds_dir, folder)):
            continue
        acts = []
        filenames = [
            int(f.replace(".png", ""))
            for f in os.listdir(os.path.join(preds_dir, folder))
            if f.endswith(".png")
        ]
        filenames = [
            os.path.join(preds_dir, folder, f"{f}.png") for f in sorted(filenames)
        ]
        for f in filenames:
            img = Image.open(f)
            # Skip black images (NSFW)
            if transforms.ToTensor()(img).sum() == 0:
                acts.append(np.nan)
                continue
            feats = feature_extractor(img)
            activation = encoder.predict(feats.cpu().detach().numpy())[0]
            acts.append(activation)
        data.append((folder, acts))
        print(f"{folder}: mean {np.nanmean(acts)}, std {np.nanstd(acts)}")
    # Save activations
    df = pd.DataFrame(data, columns=["folder", "activations"])
    df.to_csv(os.path.join(preds_dir, "activations.csv"), index=False)
