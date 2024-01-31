from model.feature_extractor import create_feature_extractor
import torch
from dataset.natural_scenes import NaturalScenesDataset
from dataset.nsd_induced import NSDInducedDataset
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
partition = 'train'
data_root = './data/'

for feature_extractor_type in  ["clip", "alexnet_1", "alexnet_2", "alexnet_3", "alexnet_4", "alexnet_5", "alexnet_6"]:
    metric = "cosine" if feature_extractor_type == "clip" else "euclidean"
    nsd = NaturalScenesDataset(data_root, subject, partition, roi='floc-faces', hemisphere='left', return_coco_id=False)
    feature_extractor = create_feature_extractor(feature_extractor_type, device=device)
    dataset = NSDInducedDataset(nsd, feature_extractor, metric, n_neighbors=0, seed=0, batch_size_feature_extraction=32)