import os
import urllib.request
import zipfile
from typing import List

import ijson
import pandas as pd
from pycocotools.coco import COCO

def download_coco_annotation_file(data_root: str):
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, "r")
    zip_file_object.extractall(path=data_root)


def coco_categories(coco_img_idxs: List[int], data_root: str):
    stim_descriptions = pd.read_csv(os.path.join(data_root, "nsd_stim_info_merged.csv"), index_col=0)

    annot_file = {
        'train2017': os.path.join(data_root, "annotations", "instances_train2017.json"),
        'val2017': os.path.join(data_root, "annotations", "instances_val2017.json"),
    }
    labels = {
        k: {l['id']: {'name': l['name'], 'supercategory': l['supercategory']} for l in (o for o in ijson.items(open(v,'r'), 'categories.item'))} for k, v in annot_file.items()
    }
    coco = {
        k: COCO(v) for k, v in annot_file.items()
    }

    categories = []
    for coco_img_idx in coco_img_idxs:
        split = stim_descriptions.iloc[coco_img_idx]['cocoSplit']
        coco_annot_IDs = coco[split].getAnnIds([stim_descriptions.iloc[coco_img_idx]["cocoId"]])
        coco_annot = coco[split].loadAnns(coco_annot_IDs)
        cat = set()
        for annot in coco_annot:
            if annot['area'] > 1000:
                cat.update([labels[split][annot['category_id']]['name'], labels[split][annot['category_id']]['supercategory']])
        categories.append(list(cat))
    return categories
