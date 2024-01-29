import numpy as np
import os.path as op
from sklearn.metrics import pairwise_distances

def distance_matrix(data_root: str, subject: int, partition: str, feature_extractor_type: str, metric: str):
    partition = partition.replace("debug_", "")
    f = op.join(data_root, f"subj{subject:02d}", ("training_split" if partition == "train" else "test_split"), 'features', f'{feature_extractor_type}.npy')
    if not op.exists(f):
        raise ValueError(f"{feature_extractor_type} features for subject {subject} and partition {partition} have not been computed yet.")
    features = np.load(f)
    D = pairwise_distances(features, metric=metric)
    return D