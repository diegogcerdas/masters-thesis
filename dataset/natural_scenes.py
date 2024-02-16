import os
from typing import List, Union

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils.nsd_utils import parse_rois, get_roi_indices, load_whole_surface, get_voxel_neighborhood


class NaturalScenesDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        subject: int,
        partition: str,
        roi: Union[str, List[str]] = None,
        center_voxel: int = None,
        n_neighbor_voxels: int = None,
        hemisphere: str = None,
        return_coco_id: bool = True,
    ):
        super().__init__()
        assert partition in ["train", "test", "debug_train", "debug_test"]
        assert subject in range(1, 9)
        if partition == "train":
            assert hemisphere is not None
            assert (roi is not None) or (center_voxel is not None and n_neighbor_voxels is not None)

        self.root = root
        self.subject = subject
        self.partition = partition
        self.return_coco_id = return_coco_id

        self.df = self.build_image_info_df()
        self.df = self.df[self.df.partition == partition.replace("debug_", "")]
        self.split = "train" if "train" in partition else "test"

        if partition == "train":
            subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")
            self.fmri_data, self.fs_indices, self.fs_coords = load_whole_surface(subj_dir, hemisphere)
            self.hemisphere = hemisphere
            self.roi_indices = []
            if roi is not None:
                if isinstance(roi, str): roi = [roi]
                roi_names, roi_classes = parse_rois(roi)
                self.roi_indices += get_roi_indices(subj_dir, roi_names, roi_classes, hemisphere)
            if center_voxel is not None and n_neighbor_voxels is not None:
                self.roi_indices += get_voxel_neighborhood(self.fs_indices, self.fs_coords, center_voxel, n_neighbor_voxels)
            self.roi_indices = sorted(list(set(self.roi_indices)))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        coco_id = self.df.iloc[idx]["coco_id"]
        img = Image.open(os.path.join(self.root, self.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img)
        if self.partition == "train":
            activation = self.fmri_data[idx, self.roi_indices]
            if self.return_coco_id:
                return img, activation, coco_id
            return img, activation
        if self.return_coco_id:
            return img, coco_id
        return img

    def build_image_info_df(self):
        subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")
        data_info_path = os.path.join(subj_dir, "image_info.csv")
        if os.path.exists(data_info_path):
            df = pd.read_csv(data_info_path)
        else:
            data = []
            training_dir = os.path.join(subj_dir, "training_split", "training_images")
            for filename in os.listdir(training_dir):
                coco_id = int(filename.split("_")[1].split(".")[0][-5:])
                filename = os.path.join(training_dir.replace(self.root, ""), filename)
                data.append([filename, "train", coco_id])
            testing_dir = os.path.join(subj_dir, "test_split", "test_images")
            for filename in os.listdir(testing_dir):
                coco_id = int(filename.split("_")[1].split(".")[0][-5:])
                filename = os.path.join(testing_dir.replace(self.root, ""), filename)
                data.append([filename, "test", coco_id])
            df = pd.DataFrame(data, columns=["filename", "partition", "coco_id"])
            df = df.sort_values(by=["partition", "filename"])
            df.to_csv(data_info_path, index=False)
        return df
