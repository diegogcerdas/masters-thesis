import json
import os

from tqdm import tqdm

from datasets.nsd import NaturalScenesDataset
from utils.coco_utils import coco_annotation


def build_inverted_index(data_root: str, subject: int):
    index = {}
    for partition in ["train", "test"]:
        dataset = NaturalScenesDataset(data_root, subject, f"debug_{partition}")
        for i in tqdm(range(len(dataset))):
            _, coco_id = dataset[i]
            _, nouns = coco_annotation(coco_id, "./data", True)
            for noun in nouns:
                index.setdefault(noun, [])
                index[noun].append(i)
        json.dump(
            index,
            open(
                os.path.join(
                    data_root, f"subj{subject:02d}/inverted_index_{partition}.json"
                ),
                "w",
            ),
        )
