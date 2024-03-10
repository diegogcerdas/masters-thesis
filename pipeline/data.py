import numpy as np

from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset


def load_dataset(
    data_dir: str,
    subject: int,
    hemisphere: str,
    roi: str,
    center_voxel: int,
    n_neighbor_voxels: int,
    voxels_filename: str,
    feature_extractor_type: str,
    distance_metric: str,
    n_neighbors: int,
    seed: int,
    device: str,
) -> NSDFeaturesDataset:
    voxel_idx = (
        np.load(voxels_filename).astype(np.int64).tolist()
        if voxels_filename is not None
        else None
    )
    nsd = NaturalScenesDataset(
        root=data_dir,
        subject=subject,
        partition="train",
        hemisphere=hemisphere,
        roi=roi,
        center_voxel=center_voxel,
        n_neighbor_voxels=n_neighbor_voxels,
        voxel_idx=voxel_idx,
    )
    dataset = NSDFeaturesDataset(
        nsd=nsd,
        feature_extractor_type=feature_extractor_type,
        predict_average=True,
        metric=distance_metric,
        n_neighbors=n_neighbors,
        seed=seed,
        device=device,
        keep_features=True,
    )
    return dataset
