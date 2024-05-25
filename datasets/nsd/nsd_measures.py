import os
import numpy as np
import torch
from torch.utils import data
from datasets.nsd.nsd import NaturalScenesDataset
from typing import List
from PIL import Image
from skimage.util import view_as_blocks
from tqdm import tqdm
from torchvision import transforms
from methods.low_level_attributes.image_measures import compute_warmth, compute_saturation, compute_brightness, compute_entropy


class NSDMeasuresDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
        measures: List[str],
        patches_shape: tuple,
        img_shape: tuple,
        predict_average: bool = True,
    ):
        super().__init__()
        self.measures = measures if isinstance(measures, list) else [measures]
        assert all([m in ["depth", "surface_normal", "gaussian_curvature", "warmth", "saturation", "brightness", "entropy"] for m in self.measures])
        assert (img_shape[0] % patches_shape[0] == 0) and (img_shape[1] % patches_shape[1] == 0)
        self.nsd = nsd
        self.patches_shape = patches_shape
        self.img_shape = img_shape
        self.patch_size = (img_shape[0]//patches_shape[0], img_shape[1]//patches_shape[1])
        self.predict_average = predict_average
        self.averages = self.compute_averages()
        self.stdevs = self.compute_stdevs()
        if nsd.return_activations:
            self.targets = self.compute_targets()
            self.target_size = 1 if predict_average else len(nsd.roi_indices)

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.nsd.root, self.nsd.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img).float()
        measures = []
        for m in self.measures:
            measures.append(self.compute_measure(idx, m, normalize=False))
        measures = np.concatenate(measures, axis=0)
        # Resize measures
        measures = torch.from_numpy(measures).float().unsqueeze(0)
        measures = torch.nn.functional.interpolate(
            measures,
            size=self.img_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        measures = measures.numpy()
        # Extract patches
        patches = view_as_blocks(measures, (measures.shape[0], self.patch_size[0], self.patch_size[1]))
        patches = torch.tensor(patches).float().mean(dim=(-1,-2)).squeeze(0).permute(2, 0, 1)
        
        if self.nsd.return_activations:
            target = self.targets[idx]
            return img, patches, target
        else: 
            return img, patches
    
    def compute_measure(self, idx, measure, normalize=True):
        f = os.path.join(self.nsd.root, self.nsd.df.iloc[idx]["filename"])
        img = Image.open(f)
        if measure == 'warmth':
            result = compute_warmth(img)
        elif measure == 'saturation':
            result = compute_saturation(img)
        elif measure == 'brightness':
            result = compute_brightness(img)
        elif measure == 'entropy':
            result = compute_entropy(img)
        else:
            f = f.replace("training_images", measure).replace(".png", ".npy")
            result = np.load(f)
        if normalize:
            result = (result - self.averages[measure]) / self.stdevs[measure]
        return result

    def compute_targets(self):
        targets = self.nsd.fmri_data[:, self.nsd.roi_indices]
        targets_mean = targets.mean(0, keepdims=True)
        targets_std = targets.std(0, keepdims=True)
        targets = (targets - targets_mean) / targets_std
        if self.predict_average:
            targets = targets.mean(1)
        return targets
    
    def compute_averages(self):
        folder = os.path.join(self.nsd.subj_dir, "averages")
        os.makedirs(folder, exist_ok=True)
        averages = {}
        missing = []
        for m in self.measures:
            f = os.path.join(folder, f"{m}.npy")
            if os.path.exists(f):
                averages[m] = np.load(f)
            else:
                missing.append(m)
        if missing:
            for m in missing:
                tmp = self.compute_measure(0, m, normalize=False)
                averages[m] = np.zeros_like(tmp)
            for i in tqdm(range(len(self.nsd)), desc=f"Computing averages..."):
                for m in missing:
                    averages[m] += self.compute_measure(i, m, normalize=False)
            for m in missing:
                averages[m] /= len(self.nsd)
                np.save(os.path.join(folder, f"{m}.npy"), averages[m])
        return averages
    
    def compute_stdevs(self):
        folder = os.path.join(self.nsd.subj_dir, "stdevs")
        os.makedirs(folder, exist_ok=True)
        stdevs = {}
        missing = []
        for m in self.measures:
            f = os.path.join(folder, f"{m}.npy")
            if os.path.exists(f):
                stdevs[m] = np.load(f)
            else:
                missing.append(m)
        if missing:
            for m in missing:
                tmp = self.compute_measure(0, m, normalize=False)
                stdevs[m] = np.zeros_like(tmp)
            for i in tqdm(range(len(self.nsd)), desc=f"Computing stdevs..."):
                for m in missing:
                    stdevs[m] += (self.compute_measure(i, m, normalize=False) - self.averages[m])**2
            for m in missing:
                stdevs[m] = np.sqrt(stdevs[m] / len(self.nsd))
                np.save(os.path.join(folder, f"{m}.npy"), stdevs[m])
        return stdevs
