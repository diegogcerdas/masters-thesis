import torch
import os
from dataset.natural_scenes import NaturalScenesDataset
from dataset.nsd_induced import NSDInducedDataset
from model.feature_extractor import create_feature_extractor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from utils.nsd_utils import plot_interactive_surface

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
partition = 'train'
data_root = '../data/'
roi = 'floc-faces'
hemisphere = 'right'

feature_extractor_type = "vae"
metric = 'euclidean'
n_neighbors = 0
seed = 0

center_voxel = 137473
n_neighbor_voxels = 100

nsd = NaturalScenesDataset(
    root=data_root, 
    subject=subject, 
    partition=partition, 
    roi=roi, 
    # center_voxel=center_voxel,
    # n_neighbor_voxels=n_neighbor_voxels,
    hemisphere=hemisphere
)

feature_extractor = create_feature_extractor(feature_extractor_type, device)
dataset = NSDInducedDataset(nsd, feature_extractor, True, metric, n_neighbors, seed)
del nsd, feature_extractor