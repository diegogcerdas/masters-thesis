import json
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from datasets.nsd.nsd import NaturalScenesDataset

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'fire', 'street', 'stop', 
                'parking', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 
                'umbrella', 'shoe', 'eye', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'kite', 'baseball', 'baseball', 
                'skateboard', 'surfboard', 'tennis', 'bottle', 'plate', 'wine', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted', 'bed', 'mirror', 
                'dining', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell', 'microwave', 'oven', 
                'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy', 'hair', 'toothbrush', 'hair']

def download_coco_annotation_file(dataset_root: str):
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, "r")
    zip_file_object.extractall(path=dataset_root)

def save_segmentation_masks(dataset_root: str, subject: int):

    stim_descriptions = pd.read_csv(os.path.join(dataset_root, "nsd_stim_info_merged.csv"), index_col=0)
    annot_file = {
        'train2017': os.path.join(dataset_root, "annotations", "instances_train2017.json"),
        'val2017': os.path.join(dataset_root, "annotations", "instances_val2017.json"),
    }
    coco = {k: COCO(v) for k, v in annot_file.items()}
    nsd = NaturalScenesDataset(dataset_root, subject, 'all')

    for i in tqdm(range(len(nsd)), desc=f"Saving COCO segmentation masks..."):
        f = os.path.join(nsd.root, nsd.df.iloc[i]["filename"])
        f = f.replace("training_images", 'segmentation').replace(".png", ".npy")
        nsd_idx = nsd.df.iloc[i]["nsd_idx"]
        
        # Get annotations for the current image
        row = stim_descriptions.iloc[nsd_idx]
        coco_id = int(row['cocoId'])
        coco_split = row['cocoSplit']
        crop_values = eval(row['cropBox'])
        annIds = coco[coco_split].getAnnIds(imgIds=coco_id, iscrowd=None)
        anns = coco[coco_split].loadAnns(annIds)

        # Initialize a mask for the whole image
        img_info = coco[coco_split].loadImgs(coco_id)[0]
        composite_mask = np.zeros((img_info['height'], img_info['width']))

        for ann in anns:
            # Generate segmentation mask for the current annotation
            mask = coco[coco_split].annToMask(ann)
            # Update the composite mask
            composite_mask = np.maximum(composite_mask, mask * ann['category_id'])

        # Crop the mask
        top, bottom, left, right = crop_values
        topCrop = int(round(composite_mask.shape[0] * top))
        bottomCrop = int(round(composite_mask.shape[0] * bottom))
        leftCrop = int(round(composite_mask.shape[1] * left))
        rightCrop = int(round(composite_mask.shape[1] * right))
        cropped_image_array = composite_mask[topCrop:composite_mask.shape[0]-bottomCrop, leftCrop:composite_mask.shape[1]-rightCrop]
        cropped_image_array = cropped_image_array.astype(np.uint8)

        # Downsample to 425x425
        cropped_image_array = Image.fromarray(cropped_image_array).resize((425, 425), Image.NEAREST)
        cropped_image_array = np.array(cropped_image_array)

        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.save(f, cropped_image_array)


def color_segmentation_mask(segmentation_mask: np.array):
    unique_labels = np.unique(segmentation_mask)
    colored_mask_array = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)  # Initialize with zeros (black)
    for label in unique_labels:
        if label > 0:  # Skip background
            color = plt.cm.jet(label / np.max(unique_labels))[:3]  # Get color from colormap
            color = (np.array(color) * 255).astype(np.uint8)  # Convert color to RGB format
            colored_mask_array[segmentation_mask == label] = color  # Apply color to label positions
    colored_mask = Image.fromarray(colored_mask_array)
    return colored_mask


def build_coco_category_search(dataset_root: str, subject: int):
    nsd = NaturalScenesDataset(dataset_root, subject, 'all')

    nsd_idx2categories = {}
    category2nsd_idxs = {}

    for i in tqdm(range(len(nsd))):
        f = os.path.join(nsd.root, nsd.df.iloc[i]["filename"])
        f = f.replace('training_images', 'segmentation').replace('.png', '.npy')
        nsd_idx = int(nsd.df.iloc[i]["nsd_idx"])
        mask = np.load(f).astype(np.uint8)
        cats, counts = np.unique(mask, return_counts=True)
        categories = []
        for cat, count in zip(cats, counts):
            percent = (count / mask.size) * 100
            if cat != 0 and percent >= 0.5:
                categories.append(COCO_CLASSES[cat-1])
        nsd_idx2categories[nsd_idx] = categories
        for c in categories:
            category2nsd_idxs.setdefault(c, [])
            category2nsd_idxs[c].append(nsd_idx)
    
    f = os.path.join(dataset_root, f"subj{subject:02d}/nsd_idx2categories.json")
    json.dump(nsd_idx2categories, open(f, "w"))
    f = os.path.join(dataset_root, f"subj{subject:02d}/category2nsd_idxs.json")
    json.dump(category2nsd_idxs, open(f, "w"))
