import os
import numpy as np
from PIL import Image
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from torch.utils import data
from datasets.nsd.nsd import NaturalScenesDataset
from methods.high_level_attributes.clip_extractor import create_clip_extractor
from torchvision import transforms


class NSDCLIPFeaturesDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
        clip_extractor_type: str,
        predict_average: bool,
    ):
        super().__init__()
        assert nsd.return_activations
        self.nsd = nsd
        self.clip_extractor_type = clip_extractor_type
        self.predict_average = predict_average
        self.features = self.load_features()
        self.D = self.load_distance_matrix()
        self.targets = self.compute_targets()
        self.target_size = 1 if predict_average else len(nsd.roi_indices)

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.nsd.root, self.nsd.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img).float()
        target = self.targets[idx]
        return img, target

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
    
    def compute_targets(self):
        targets = self.nsd.fmri_data[:, self.nsd.roi_indices]
        targets_mean = targets.mean(0, keepdims=True)
        targets_std = targets.std(0, keepdims=True)
        targets = (targets - targets_mean) / targets_std
        if self.predict_average:
            targets = targets.mean(1)
        return targets

    def compute_low_dim(self, seed):
        print("Computing low dimensional representation...")
        tsne = manifold.TSNE(n_components=2, metric="precomputed", init="random", random_state=seed)
        low_dim = tsne.fit_transform(self.D)
        print("Done.")
        return low_dim
