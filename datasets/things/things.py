import os
from typing import List, Union

import numpy as np
import pandas as pd
from mat73 import loadmat
from PIL import Image
from torch.utils import data
from torchvision import transforms


class ThingsDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        subject_folder: int,
        partition: str,
        roi: Union[str, List[str]],
        electrode_idx: List[int] = None,
    ):
        super().__init__()
        assert partition in ["train", "test"]
        assert subject_folder in ['monkeyN', 'monkeyF']

        if partition == "test":
            raise NotImplementedError("Test partition not implemented yet. Missing images.")

        self.root = root

        self.index = pd.read_csv(os.path.join(self.root, "index.csv"))
        self.activations = loadmat(os.path.join(self.root, subject_folder, "THINGS_normMUA.mat"))
        self.activations = self.activations['train_MUA' if partition == 'train' else 'test_MUA'].T

        if electrode_idx is not None:
            self.roi_indices = electrode_idx
        else:
            with open(os.path.join(self.root, subject_folder, 'rois.txt'), 'r') as f:
                rois = np.array([line.strip() for line in f])
            self.roi_indices = np.where(rois == roi)[0]
            assert len(self.roi_indices) > 0, f"ROI {roi} not found for subject {subject_folder}"

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.index.iloc[idx]["things_path"]))
        img = transforms.ToTensor()(img).float()
        activation = self.activations[idx][self.roi_indices]
        return img, activation