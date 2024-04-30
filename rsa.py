import torch
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.img_utils import save_images
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from utils.nsd_utils import parse_rois, get_roi_indices
import io
from scipy.stats import spearmanr

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
data_root = './data/NSD'
roi = [
    "PPA",
]
hemisphere = 'right'
seed = 0

nsd = NaturalScenesDataset(
    root=data_root,
    subject=subject,
    partition="train",
    hemisphere=hemisphere,
    roi=roi,
)

feature_extractor_type = "clip_2_0"
metric = 'cosine'
n_neighbors = 0

dataset = NSDFeaturesDataset(
    nsd=nsd,
    feature_extractor_type=feature_extractor_type,
    predict_average=False,
    metric=metric,
    n_neighbors=n_neighbors,
    seed=seed,
    device=device,
    keep_features=True,
)
del nsd

# Load index and inverted index
f = os.path.join(data_root, f'subj{subject:02d}/index_train.json')
index = json.load(open(f, 'r'))
f = os.path.join(data_root, f'subj{subject:02d}/inverted_index_train.json')
inv_index = json.load(open(f, 'r'))

subset = None

subsets={
    'wild_animals': {
        'all_positives': [['animal']],
        'negatives': ['bird', 'cat', 'dog', 'person', 'vehicle'],
    },
    'people_sports': {
        'all_positives': [['person', 'sports']],
        'negatives': ['animal', 'vehicle', 'food'],
    },
    'pets': {
        'all_positives': [['cat'], ['dog']],
        'negatives': ['person'],
    },
    'vehicles': {
        'all_positives': [['vehicle']],
        'negatives': ['animal', 'person'],
    },
    'food': {
        'all_positives': [['food']],
        'negatives': ['animal', 'person'],
    },
}

if subset is None:
    all_indices = np.arange(len(dataset))

else:
    all_positives = subsets[subset]['all_positives']
    negatives = subsets[subset]['negatives']

    remaining_negatives = {}
    all_indices = set()
    for positives in all_positives:
        pos = set(inv_index[positives[0]])
        for positive in positives:
            pos = set.intersection(pos, set(inv_index[positive])) 
        for p in pos:
            categories = index[str(p)]
            rem = []
            keep = True
            for c in categories:
                if c in negatives:
                    keep = False
                elif c not in positives:
                    rem.append(c)
            if keep:
                all_indices.add(p)
                for r in rem:
                    remaining_negatives.setdefault(r, 0)
                    remaining_negatives[r] += 1
    all_indices = np.array(list(all_indices))


roi_axis = dataset.targets.mean(axis=1).numpy()
model = LinearRegression(fit_intercept=False).fit(roi_axis.reshape(-1, 1), dataset.features - dataset.features.mean(axis=0))
directiona1_vector = model.coef_[:,0] / np.linalg.norm(model.coef_[:,0])
directiona1_vector = torch.from_numpy(directiona1_vector).to(device)

voxel_D = pairwise_distances(dataset.targets, metric='euclidean')

df = pd.read_csv('attributes.tsv', sep='\t')
subspace = np.empty((len(dataset), len(df)))
for i, description in enumerate(df['description']):
    idx = df[df.description == description].index[0]
    x0 = np.load(f'attribute_vectors/{idx}_0.npy')
    x1 = np.load(f'attribute_vectors/{idx}_1.npy')
    shift_vector = x1 - x0
    shift = 1 - pairwise_distances(dataset.features, shift_vector.reshape(1, -1), metric='cosine')[all_indices][:, 0]
    subspace[:, i] = shift
subspace_D = pairwise_distances(subspace, metric='euclidean')

idx = torch.triu_indices(*voxel_D.shape, offset=1)
RDM1 = voxel_D[idx[0], idx[1]]
RDM2 = dataset.D[idx[0], idx[1]]
r = spearmanr(RDM1, RDM2).statistic
print(r)

idx = torch.triu_indices(*voxel_D.shape, offset=1)
RDM1 = voxel_D[idx[0], idx[1]]
RDM2 = subspace_D[idx[0], idx[1]]
r = spearmanr(RDM1, RDM2).statistic
print(r)

idx = torch.triu_indices(*voxel_D.shape, offset=1)
RDM1 = dataset.D[idx[0], idx[1]]
RDM2 = subspace_D[idx[0], idx[1]]
r = spearmanr(RDM1, RDM2).statistic
print(r)