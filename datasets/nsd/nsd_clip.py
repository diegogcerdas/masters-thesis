import os

import numpy as np
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from torch.utils import data

from datasets.nsd.nsd import NaturalScenesDataset
from methods.high_level_attributes.clip_extractor import create_clip_extractor


class NSDCLIPFeaturesDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
        clip_extractor_type: str,
        predict_average: bool,
    ):
        super().__init__()
        self.nsd = nsd
        self.clip_extractor_type = clip_extractor_type
        self.predict_average = predict_average
        indices = list(self.nsd.df.index)
        self.features = self.load_features()[indices, :]
        self.D = self.load_distance_matrix()[indices, :][:, indices]

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        features = self.features[idx]
        if self.nsd.return_activations:
            img, activation, _ = self.nsd[idx]
            return img, features, activation
        img, _ = self.nsd[idx]
        return img, features

    def load_features(self):
        folder = os.path.join(self.nsd.root, f"subj{self.nsd.subject:02d}", "training_split", "features")
        f = os.path.join(folder, f"{self.clip_extractor_type}.npy")
        if not os.path.exists(f):
            print("Computing features...")
            clip_extractor = create_clip_extractor(self.clip_extractor_type)
            features = clip_extractor.extract_for_dataset(self.nsd)
            os.makedirs(folder, exist_ok=True)
            np.save(f, features)
            print("Done.")
        else:
            features = np.load(f).astype(np.float32)
        return features

    def load_distance_matrix(self):
        folder = os.path.join(self.nsd.root, f"subj{self.nsd.subject:02d}", "training_split", "distance")
        f = os.path.join(folder, f"{self.clip_extractor_type}.npy")
        if not os.path.exists(f):
            print("Computing distance matrix...")
            D = pairwise_distances(self.features, metric='cosine').astype(np.float32)
            os.makedirs(folder, exist_ok=True)
            np.save(f, D)
            print("Done.")
        else:
            D = np.load(f).astype(np.float32)
        return D

    def compute_low_dim(self, seed):
        print("Computing low dimensional representation...")
        tsne = manifold.TSNE(n_components=2, metric="precomputed", init="random", random_state=seed)
        low_dim = tsne.fit_transform(self.D)
        print("Done.")
        return low_dim
