import os

import numpy as np
import torch
from PIL import Image
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset.natural_scenes import NaturalScenesDataset


class NSDInducedDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
        feature_extractor,
        predict_average: bool,
        metric: str,
        n_neighbors: int,
        seed: int,
        batch_size_feature_extraction: int = 32,
    ):
        super().__init__()
        assert nsd.partition == "train"
        self.nsd = nsd
        self.predict_average = predict_average
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.batch_size_feature_extraction = batch_size_feature_extraction
        self.D = self.load_distance_matrix(feature_extractor, metric)
        self.targets, self.targets_mean, self.targets_std = self.compute_targets(self.D)
        self.target_size = 1 if predict_average else len(nsd.roi_indices)
        self.low_dim = self.compute_low_dim(self.D)

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.nsd.root, self.nsd.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img)
        target = self.targets[idx].squeeze().float()
        low_dim = self.low_dim[idx]
        return img, target, low_dim

    def load_distance_matrix(self, feature_extractor, metric):
        folder = os.path.join(
            self.nsd.root, f"subj{self.nsd.subject:02d}", "training_split", "distance"
        )
        f = os.path.join(folder, f"{feature_extractor.name}_{metric}.npy")
        if not os.path.exists(f):
            print("Computing distance matrix...")
            self.nsd.partition = "debug_train"
            features = feature_extractor.extract_for_dataset(
                self.nsd, self.batch_size_feature_extraction
            )
            self.nsd.partition = "train"
            D = pairwise_distances(features, metric=metric).astype(np.float32)
            os.makedirs(folder, exist_ok=True)
            np.save(f, D)
            print("Done.")
        else:
            D = np.load(f)
        return D

    def compute_targets(self, distance_matrix):
        targets = []
        for i in tqdm(range(len(self)), desc="Computing targets..."):
            closest = np.argsort(distance_matrix[i, :])[: self.n_neighbors + 1]
            acts = self.nsd.fmri_data[closest, self.nsd.roi_indices].mean(axis=0)
            if self.predict_average:
                acts = acts.mean().item()
            targets.append(acts)
        targets = (
            torch.tensor(targets) if self.predict_average else torch.stack(targets)
        )
        targets_mean = targets.mean()
        targets_std = targets.std()
        targets = (targets - targets_mean) / targets_std
        return targets, targets_mean, targets_std

    def compute_low_dim(self, distance_matrix):
        print("Computing low dimensional representation...")
        tsne = manifold.TSNE(
            n_components=2, metric="precomputed", init="random", random_state=self.seed
        )
        low_dim = tsne.fit_transform(distance_matrix)
        print("Done.")
        return low_dim
