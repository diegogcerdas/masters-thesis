import json
import os
from tqdm import tqdm
from datasets.nsd import NaturalScenesDataset
from utils.coco_utils import coco_categories


def build_index(data_root: str, subject: int):
    for partition in ["train", "test"]:
        dataset = NaturalScenesDataset(data_root, subject, f"debug_{partition}")
        coco_img_idxs = dataset.df['coco_img_idx'].values
        categories = coco_categories(coco_img_idxs, data_root)
        index = {i: c for i, c in enumerate(categories)}
        json.dump(
            index,
            open(
                os.path.join(
                    data_root, f"subj{subject:02d}/index_{partition}.json"
                ),
                "w",
            ),
        )

def build_inverted_index(data_root: str, subject: int):
    for partition in ["train", "test"]:
        index = {}
        dataset = NaturalScenesDataset(data_root, subject, f"debug_{partition}")
        coco_img_idxs = dataset.df['coco_img_idx'].values
        categories = coco_categories(coco_img_idxs, data_root)
        for i, cat in tqdm(enumerate(categories)):
            for c in cat:
                index.setdefault(c, [])
                index[c].append(i)
        json.dump(
            index,
            open(
                os.path.join(
                    data_root, f"subj{subject:02d}/inverted_index_{partition}.json"
                ),
                "w",
            ),
        )
