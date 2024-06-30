import os
import pathlib
from typing import List, Union
import re

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms

from datasets.nsd.utils.nsd_utils import (get_roi_indices,
                                          get_voxel_neighborhood,
                                          load_whole_surface, parse_rois,
                                          get_subset_indices)


class NaturalScenesDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        subject: int,
        partition: str,
        transform: transforms.Compose = None,
        hemisphere: str = None,
        roi: Union[str, List[str]] = None,
        center_voxel: int = None,
        n_neighbor_voxels: int = None,
        voxel_idx: List[int] = None,
        return_average: bool = False,
        subset: str = None,
    ):
        super().__init__()
        assert partition in ["train", "test", "all"]
        assert subject in range(1, 9)
        self.return_activations = (roi is not None) or (center_voxel is not None) or (voxel_idx is not None)

        self.root = pathlib.Path(root)
        self.subject = subject
        self.partition = partition
        self.transform = transform
        self.return_average = return_average
        self.subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")

        self.df = self.build_image_info_df()
        if partition != "all":
            self.df = self.df[self.df.partition == partition]
        
        if subset is not None:
            self.df = self.df[self.df.nsd_idx.isin(get_subset_indices(self, subset))]

        if self.return_activations:
            indices = self.df["index"].values
            self.fmri_data, self.fs_indices, self.fs_coords = load_whole_surface(
                self.subj_dir, hemisphere
            )
            self.fmri_data = self.fmri_data[indices, :]
            self.hemisphere = hemisphere
            self.roi_indices = []
            if roi is not None:
                if roi == "all":
                    self.roi_indices = list(range(self.fmri_data.shape[1]))
                else:
                    if isinstance(roi, str):
                        roi = [roi]
                    roi_names, roi_classes = parse_rois(roi)
                    self.roi_indices += get_roi_indices(
                        self.subj_dir, roi_names, roi_classes, hemisphere
                    )
            if center_voxel is not None and n_neighbor_voxels is not None:
                self.roi_indices += get_voxel_neighborhood(
                    self.fs_indices, self.fs_coords, center_voxel, n_neighbor_voxels
                )
            if voxel_idx is not None:
                assert all([idx in self.fs_indices for idx in voxel_idx])
                self.roi_indices += voxel_idx
            assert len(self.roi_indices) > 0
            self.roi_indices = sorted(list(set(self.roi_indices)))
            self.activations = self.fmri_data[:, self.roi_indices]
            if return_average:
                self.activations = self.activations.mean(axis=1)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.df.iloc[idx]["filename"])).convert("RGB")
        if self.transform:
            img = self.transform(img).float()
        nsd_idx = self.df.iloc[idx]["nsd_idx"]
        if self.return_activations:
            activation = self.activations[idx]
            return img, activation, nsd_idx
        return img, nsd_idx

    def build_image_info_df(self):
        with open(os.path.join(self.root, 'algonauts_shared872.txt'), 'r') as file:
            algonauts_shared872 = [int(l.replace('\n', '')) for l in file.readlines()]
        if not os.path.exists(self.subj_dir):
            raise ValueError(f"Data for subject {self.subject} not found in {self.root}.")
        data_info_path = os.path.join(self.subj_dir, "image_info.csv")
        if os.path.exists(data_info_path):
            df = pd.read_csv(data_info_path)
        else:
            data = []
            training_dir = os.path.join(f"subj{self.subject:02d}", "training_split", "training_images")
            for filename in os.listdir(os.path.join(self.root, training_dir)):
                nsd_idx = int(filename.split("_")[1].split(".")[0][-5:])
                img_idx = int(re.search('{}(.+?){}'.format('train-', '_nsd'), filename).group(1)) - 1
                filename = os.path.join(training_dir, filename)
                partition = "test" if nsd_idx in algonauts_shared872 else "train"
                data.append([img_idx, filename, partition, nsd_idx])
            df = pd.DataFrame(data, columns=["index", "filename", "partition", "nsd_idx"])
            df = df.sort_values(by=["index"])
            df.to_csv(data_info_path, index=False)
        return df
