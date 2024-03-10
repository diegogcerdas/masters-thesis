import os

import numpy as np
from PIL import Image

from datasets.nsd_features import NSDFeaturesDataset
from utils.img_utils import save_images


def save_stimuli(
    low_std: float,
    high_std: float,
    num_stimuli: int,
    targets: np.ndarray,
    dataset: NSDFeaturesDataset,
    save_dir: str,
):
    idx = np.where(
        (targets > np.mean(targets) + low_std * np.std(targets))
        & (targets < np.mean(targets) + high_std * np.std(targets))
    )[0]
    idx_sample = np.random.choice(
        idx,
        min(len(idx), num_stimuli),
        replace=False,
    )
    images = [
        Image.open(os.path.join(dataset.nsd.root, dataset.nsd.df.iloc[idx]["filename"]))
        for idx in idx_sample
    ]
    print(f"Total activation mean: {np.mean(targets)}, std: {np.std(targets)}.")
    print(f"Number of stimuli: {len(idx)}, sampled {len(idx_sample)}.")
    print(
        f"Sampled activation mean: {np.mean(targets[idx_sample])}, std {np.std(targets[idx_sample])}."
    )

    # Save Max-EIs
    save_images(images, save_dir)
