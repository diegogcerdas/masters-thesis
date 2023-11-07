import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class NaturalScenesDataset(Dataset):
    def __init__(
        self,
        root: str,
        subject: int,
        partition: str,
        roi: str = None,
        hemisphere: str = None,
    ):
        super().__init__()
        assert partition in ["train", "test"]
        # TODO: Add support for multiple subjects
        assert subject in range(1, 9)
        assert partition == "test" or ((roi is not None) and (hemisphere is not None))
        self.root = root
        self.subject = subject
        self.partition = partition

        self.df = self.build_image_info_df()
        self.df = self.df[self.df.partition == partition]

        if partition == "train":
            # TODO: Add support for multiple ROIs
            self.roi, self.roi_class = self.parse_roi(roi)
            self.hemisphere = hemisphere
            self.fmri_data = self.load_fmri_data()
            self.num_voxels = self.fmri_data.shape[1]

        if partition == "train":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(224, antialias=True),
                    transforms.Lambda(lambda x: x * np.random.uniform(0.95, 1.05)),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                    RandomSpatialOffset(offset=4),
                    transforms.Lambda(
                        lambda x: x + (torch.randn(x.shape) * 0.05**2).to(x.device)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(224, antialias=True),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["filename"])
        img = self.transform(img)
        if self.partition == "train":
            fmri = self.fmri_data[idx]
            return img, fmri
        return img

    def load_fmri_data(self):
        subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")
        fmri_dir = os.path.join(subj_dir, "training_split", "training_fmri")
        roi_class_dir = os.path.join(
            subj_dir,
            "roi_masks",
            f"{self.hemisphere[0]}h.{self.roi_class}_challenge_space.npy",
        )
        roi_map_dir = os.path.join(
            subj_dir, "roi_masks", f"mapping_{self.roi_class}.npy"
        )
        fmri = np.load(
            os.path.join(fmri_dir, f"{self.hemisphere[0]}h_training_fmri.npy")
        )
        roi_class = np.load(roi_class_dir)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(self.roi)]
        roi = np.asarray(roi_class == roi_mapping, dtype=int)
        fmri = fmri[:, np.where(roi)[0]]
        return torch.from_numpy(fmri).float()

    def parse_roi(self, roi):
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = "prf-visualrois"
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = "floc-bodies"
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = "floc-faces"
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = "floc-places"
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = "floc-words"
        elif roi in [
            "early",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ]:
            roi_class = "streams"
        else:
            raise ValueError("Invalid ROI")
        return roi, roi_class

    def build_image_info_df(self):
        subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")
        data_info_path = os.path.join(subj_dir, "image_info.csv")
        if os.path.exists(data_info_path):
            df = pd.read_csv(data_info_path)
        else:
            data = []
            training_dir = os.path.join(subj_dir, "training_split", "training_images")
            for filename in os.listdir(training_dir):
                filename = os.path.join(training_dir, filename)
                data.append([filename, "train"])
            testing_dir = os.path.join(subj_dir, "test_split", "test_images")
            for filename in os.listdir(testing_dir):
                filename = os.path.join(testing_dir, filename)
                data.append([filename, "test"])
            df = pd.DataFrame(data, columns=["filename", "partition"])
            df = df.sort_values(by=["partition", "filename"])
            df.to_csv(data_info_path, index=False)
        return df


class RandomSpatialOffset:
    def __init__(self, offset, padding_mode="replicate"):
        self.offset = offset
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]

        dx = random.randint(-self.offset, self.offset)
        dy = random.randint(-self.offset, self.offset)

        # Perform edge value padding
        left, top, right, bottom = max(0, dx), max(0, dy), max(0, -dx), max(0, -dy)
        img = F.pad(img, (left, top, right, bottom), mode=self.padding_mode)

        # Crop back to original size
        left, top = max(0, -dx), max(0, -dy)
        img = TF.crop(img, top, left, h, w)

        return img
