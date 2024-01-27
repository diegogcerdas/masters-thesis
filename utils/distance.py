import numpy as np
from model.feature_extractor import FeatureExtractor
from dataset.natural_scenes import NaturalScenesDataset
import os.path as op
from sklearn.metrics import pairwise_distances
import os

def save_features(data_root: str, subject: int, feature_extractor: FeatureExtractor, batch_size: int):
    for partition in ["train", "test"]:
        f = op.join(data_root, f"subj{subject:02d}", ("training_split" if partition == "train" else "test_split"), 'features', f'{feature_extractor.name}.npy')
        if op.exists(f):
            print(f"{feature_extractor.name} features for subject {subject} and partition {partition} already exist. Skipping.")
            continue
        dataset = NaturalScenesDataset(data_root, subject, f"debug_{partition}")
        features = feature_extractor.extract_for_dataset(dataset, batch_size)
        folder = op.join(data_root, f"subj{subject:02d}", ("training_split" if partition == "train" else "test_split"), 'features')
        os.makedirs(folder, exist_ok=True)
        np.save(op.join(folder, f'{feature_extractor.name}.npy'), features)

def distance_matrix(data_root: str, subject: int, partition: str, feature_extractor: FeatureExtractor, metric: str):
    partition = partition.replace("debug_", "")
    f = op.join(data_root, f"subj{subject:02d}", ("training_split" if partition == "train" else "test_split"), 'features', f'{feature_extractor.name}.npy')
    if not op.exists(f):
        raise ValueError(f"{feature_extractor.name} features for subject {subject} and partition {partition} have not been computed yet.")
    features = np.load(f)
    return pairwise_distances(features, metric=metric)