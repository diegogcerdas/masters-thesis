import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.feature_extractor import create_feature_extractor
from utils.distance import distance_matrix

import os.path as op

from sklearn import manifold
from tqdm import tqdm

from torch.utils import data


class NaturalScenesDataset(Dataset):
    def __init__(
        self,
        root: str,
        subject: int,
        partition: str,
        roi: Union[str, List[str]] = None,
        hemisphere: str = None,
        feature_extractor_type: str = None,
        n_neighbors: int = None,
        distance_metric: str = None,
        seed: int = 42,
        batch_size_target_extraction: int = 8,
        device: str = None,
    ):
        super().__init__()
        assert partition in ["train", "test", "debug_train", "debug_test"]
        assert subject in range(1, 9)
        if partition == "train":
            assert roi is not None
            assert hemisphere is not None
            assert feature_extractor_type is not None
            assert n_neighbors >= 0
            assert distance_metric is not None

        self.root = root
        self.subject = subject
        self.partition = partition
        self.device = device

        self.df = self.build_image_info_df()
        self.df = self.df[self.df.partition == partition.replace("debug_", "")]
        self.split = "train" if "train" in partition else "test"

        if partition == "train":
            self.feature_extractor_type = feature_extractor_type
            self.feature_extractor = create_feature_extractor(feature_extractor_type, device)
            self.n_neighbors = n_neighbors
            self.metric = distance_metric
            self.seed = seed
            self.batch_size_target_extraction = batch_size_target_extraction
            self.rois, self.roi_classes = self.parse_roi(roi)
            self.hemisphere = hemisphere
            self.fmri_data = self.load_fmri_data()
            self.features, self.targets, self.low_dim = self.compute_features_targets_lowdim()
            self.input_size = self.features.shape[1]
            self.target_size = 1

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        coco_id = self.df.iloc[idx]["coco_id"]
        img = Image.open(os.path.join(self.root, self.df.iloc[idx]["filename"]))
        img = transforms.ToTensor()(img)
        if self.partition == "train":
            # TODO: add data augmentation to features?
            features = self.features[idx]
            target = self.targets[idx]
            low_dim = self.low_dim[idx]
            return features, target, low_dim
        return img, coco_id
    
    def compute_features_targets_lowdim(self):
        features = self.compute_features()
        features = (features - features.min()) / (features.max() - features.min())
        D = distance_matrix(self.root, self.subject, self.partition, self.feature_extractor, self.metric)
        model = manifold.TSNE(n_components=2, metric="precomputed", init="random", random_state=self.seed)
        targets = self.compute_targets(D)
        low_dim = model.fit_transform(D)
        return features, targets, low_dim
    
    def compute_features(self):
        folder = op.join(self.root, f"subj{self.subject:02d}", ("training_split" if self.partition == "train" else "test_split"), 'features')
        f = op.join(folder, f'{self.feature_extractor_type}.npy')
        if op.exists(f):
            d = NaturalScenesDataset(self.root, self.subject, f'debug_{self.partition}')
            dataloader = data.DataLoader(
                d,
                batch_size=self.batch_size_target_extraction,
                drop_last=False,
                shuffle=False,
            )
            features = []
            for batch in tqdm(dataloader, total=len(dataloader)):
                x = batch[0]
                x = x.to(self.device)
                bs = x.shape[0]
                x = self.feature_extractor(x).reshape((bs, -1)).detach().cpu().numpy()
                features.append(x)
            features = np.concatenate(features, axis=0)
            os.makedirs(folder, exist_ok=True)
            np.save(f, features)
        else:
            features = np.load(f)
        return features

    def compute_targets(self, D):
        folder = op.join(self.root, f"subj{self.subject:02d}", ("training_split" if self.partition == "train" else "test_split"), 'targets')
        f = op.join(folder, f'{self.feature_extractor_type}_{self.metric}_{self.n_neighbors}_{self.seed}_{self.hemisphere[0]}_{"_".join(sorted(self.rois))}.npy')
        if not op.exists(f):
            targets = []
            for i in tqdm(range(len(self))):
                closest = np.argsort(D[i, :])[:self.n_neighbors+1]
                acts = []
                for c in closest:
                    _, a, _ = self[c]
                    acts.append(a.item())
                targets.append(np.mean(acts))
            targets = np.array(targets)
            os.makedirs(folder, exist_ok=True)
            np.save(f, targets)
        else:
            targets = np.load(f)
        return targets

    def load_fmri_data(self):
        subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")
        fmri_dir = os.path.join(subj_dir, "training_split", "training_fmri")
        fmri = np.load(
            os.path.join(fmri_dir, f"{self.hemisphere[0]}h_training_fmri.npy")
        )
        fmri_mask = np.zeros(fmri.shape[1])
        for roi_, roi_class_ in zip(self.rois, self.roi_classes):
            roi_class_dir = os.path.join(
                subj_dir,
                "roi_masks",
                f"{self.hemisphere[0]}h.{roi_class_}_challenge_space.npy",
            )
            roi_map_dir = os.path.join(
                subj_dir, "roi_masks", f"mapping_{roi_class_}.npy"
            )
            roi_class_npy = np.load(roi_class_dir)
            roi_map = np.load(roi_map_dir, allow_pickle=True).item()
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi_)]
            fmri_mask += np.asarray(roi_class_npy == roi_mapping, dtype=int)
        fmri = fmri[:, np.where(fmri_mask)[0]]
        fmri = fmri.mean(-1)
        fmri = (fmri - fmri.mean()) / fmri.std()
        return torch.from_numpy(fmri).float()

    def parse_roi(self, roi):
        if roi == "prf-visualrois":
            roi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
            roi_class = ["prf-visualrois"] * len(roi)
            return roi, roi_class
        elif roi == "floc-bodies":
            roi = ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]
            roi_class = ["floc-bodies"] * len(roi)
            return roi, roi_class
        elif roi == "floc-faces":
            roi = ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]
            roi_class = ["floc-faces"] * len(roi)
            return roi, roi_class
        elif roi == "floc-places":
            roi = ["OPA", "PPA", "RSC"]
            roi_class = ["floc-places"] * len(roi)
            return roi, roi_class
        elif roi == "floc-words":
            roi = ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
            roi_class = ["floc-words"] * len(roi)
            return roi, roi_class
        elif roi == "streams":
            roi = [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]
            roi_class = ["streams"] * len(roi)
            return roi, roi_class
        roi_class = []
        for roi_ in roi:
            if roi_ in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
                roi_class.append("prf-visualrois")
            elif roi_ in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
                roi_class.append("floc-bodies")
            elif roi_ in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
                roi_class.append("floc-faces")
            elif roi_ in ["OPA", "PPA", "RSC"]:
                roi_class.append("floc-places")
            elif roi_ in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
                roi_class.append("floc-words")
            elif roi_ in [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]:
                roi_class.append("streams")
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
                coco_id = int(filename.split("_")[1].split(".")[0][-5:])
                filename = os.path.join(training_dir.replace(self.root, ''), filename)
                data.append([filename, "train", coco_id])
            testing_dir = os.path.join(subj_dir, "test_split", "test_images")
            for filename in os.listdir(testing_dir):
                coco_id = int(filename.split("_")[1].split(".")[0][-5:])
                filename = os.path.join(testing_dir.replace(self.root, ''), filename)
                data.append([filename, "test", coco_id])
            df = pd.DataFrame(data, columns=["filename", "partition", "coco_id"])
            df = df.sort_values(by=["partition", "filename"])
            df.to_csv(data_info_path, index=False)
        return df
