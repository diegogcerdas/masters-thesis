import os

import numpy as np
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from torch.utils import data
from tqdm import tqdm

from dataset.natural_scenes import NaturalScenesDataset

from PIL import Image
from torchvision import transforms


class NSDInducedDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
        feature_extractor_type: str,
        metric: str,
        n_neighbors: int,
        seed: int,
    ):
        super().__init__()
        assert nsd.partition == "train"
        self.nsd = nsd
        self.feature_extractor_type = feature_extractor_type
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.target_size = 1
        features = self.load_features()
        D = pairwise_distances(features, metric=metric)
        self.targets = self.compute_targets(D)
        self.low_dim = self.compute_low_dim(D)

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.nsd.root, self.nsd.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img)
        target = self.targets[idx]
        low_dim = self.low_dim[idx]
        return img, target, low_dim

    def load_features(self):
        folder = os.path.join(
            self.nsd.root, f"subj{self.nsd.subject:02d}", "training_split", "features"
        )
        f = os.path.join(folder, f"{self.feature_extractor_type}.npy")
        try:
            print(f"Loading features from {f}")
            features = np.load(f)
        except:
            raise ValueError(
                f"Features not found at {f}. Please run the feature extractor first."
            )
        return features

    def compute_targets(self, distance_matrix):
        folder = os.path.join(
            self.nsd.root, f"subj{self.nsd.subject:02d}", "training_split", "targets"
        )
        f = os.path.join(
            folder,
            f'{self.feature_extractor_type}_{self.metric}_{self.n_neighbors}_{self.seed}_{self.nsd.hemisphere[0]}_{"_".join(sorted(self.nsd.rois))}.npy',
        )
        if not os.path.exists(f):
            targets = []
            for i in tqdm(range(len(self)), desc="Computing targets..."):
                closest = np.argsort(distance_matrix[i, :])[: self.n_neighbors + 1]
                acts = []
                for c in closest:
                    _, a, _ = self.nsd[c]
                    acts.append(a.mean().item())
                targets.append(np.mean(acts))
            targets = np.array(targets)
            targets = (targets - targets.min()) / (targets.max() - targets.min())
            os.makedirs(folder, exist_ok=True)
            np.save(f, targets)
        else:
            targets = np.load(f)
        return targets

    def compute_low_dim(self, distance_matrix):
        print("Computing low dimensional representation...")
        tsne = manifold.TSNE(
            n_components=2, metric="precomputed", init="random", random_state=self.seed
        )
        low_dim = tsne.fit_transform(distance_matrix)
        print("Done.")
        return low_dim
