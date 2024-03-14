import torch
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
partition = 'train'
data_root = './data/'
roi = 'floc-faces'
hemisphere = 'right'

feature_extractor_type = "vae_1_5"
metric = 'euclidean'
n_neighbors = 0
seed = 0
nsd = NaturalScenesDataset(
    root=data_root, 
    subject=subject, 
    partition=partition, 
    roi=roi, 
    hemisphere=hemisphere
)
dataset = NSDFeaturesDataset(nsd, feature_extractor_type, True, metric, n_neighbors, seed, device)
del nsd, dataset

feature_extractor_type = "vae_2_0"
metric = 'euclidean'
n_neighbors = 0
seed = 0
nsd = NaturalScenesDataset(
    root=data_root, 
    subject=subject, 
    partition=partition, 
    roi=roi, 
    hemisphere=hemisphere
)
dataset = NSDFeaturesDataset(nsd, feature_extractor_type, True, metric, n_neighbors, seed, device)
del nsd, dataset

feature_extractor_type = "vae_1_5"
metric = 'cosine'
n_neighbors = 0
seed = 0
nsd = NaturalScenesDataset(
    root=data_root, 
    subject=subject, 
    partition=partition, 
    roi=roi, 
    hemisphere=hemisphere
)
dataset = NSDFeaturesDataset(nsd, feature_extractor_type, True, metric, n_neighbors, seed, device)
del nsd, dataset

feature_extractor_type = "vae_2_0"
metric = 'cosine'
n_neighbors = 0
seed = 0
nsd = NaturalScenesDataset(
    root=data_root, 
    subject=subject, 
    partition=partition, 
    roi=roi, 
    hemisphere=hemisphere
)
dataset = NSDFeaturesDataset(nsd, feature_extractor_type, True, metric, n_neighbors, seed, device)
del nsd, dataset