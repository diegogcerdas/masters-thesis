import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from nilearn import datasets
from nilearn.surface import load_surf_mesh
from tqdm import tqdm


def load_whole_surface(data_dir, hemisphere, dtype=np.float32):
    # Load fMRI data
    fmri_dir = os.path.join(data_dir, "training_split", "training_fmri")
    fmri = np.load(
        os.path.join(fmri_dir, f"{hemisphere[0]}h_training_fmri.npy")
    ).astype(dtype)
    fmri = torch.from_numpy(fmri).float()
    # Get fsaverage coordinates
    roi_dir = os.path.join(
        data_dir, "roi_masks", f"{hemisphere[0]}h.all-vertices_fsaverage_space.npy"
    )
    fsaverage_all_vertices = np.load(roi_dir).astype(np.int8)
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    surf_mesh = fsaverage[f"flat_{hemisphere}"]
    fs_coords = load_surf_mesh(surf_mesh)[0][:, :2]
    # Get original voxel indices
    fs_indices = np.arange(len(fs_coords))[np.where(fsaverage_all_vertices)[0]]
    return fmri, fs_indices, fs_coords


def get_roi_indices(data_dir, roi_names, roi_classes, hemisphere):
    indices = set()
    for roi, roi_class in zip(roi_names, roi_classes):
        # challenge_roi_class has ROI-class-specific ROI indices for all fMRI array elements
        challenge_roi_class_dir = os.path.join(
            data_dir,
            "roi_masks",
            hemisphere[0] + "h." + roi_class + "_challenge_space.npy",
        )
        challenge_roi_class = np.load(challenge_roi_class_dir)
        # roi_map is a map from ROI indices to ROI names
        roi_map_dir = os.path.join(
            data_dir, "roi_masks", "mapping_" + roi_class + ".npy"
        )
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        # roi_mapping has the indices of the specified ROI
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        # Same as challenge_roi but now a binary mask for the specified ROI
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
        indices_roi = np.where(challenge_roi)[0]
        indices.update(indices_roi)
    return sorted(list(indices))


def parse_rois(rois):
    roi_names = []
    roi_classes = []
    for roi in rois:
        if roi == "prf-visualrois":
            roi_names += ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
            roi_classes += ["prf-visualrois"] * 7
        elif roi == "floc-bodies":
            roi_names += ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]
            roi_classes += ["floc-bodies"] * 4
        elif roi == "floc-faces":
            roi_names += ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]
            roi_classes += ["floc-faces"] * 5
        elif roi == "floc-places":
            roi_names += ["OPA", "PPA", "RSC"]
            roi_classes += ["floc-places"] * 3
        elif roi == "floc-words":
            roi_names += ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
            roi_classes += ["floc-words"] * 5
        elif roi == "streams":
            roi_names += [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]
            roi_classes += ["streams"] * 7
        elif roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_names.append(roi)
            roi_classes.append("prf-visualrois")
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_names.append(roi)
            roi_classes.append("floc-bodies")
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_names.append(roi)
            roi_classes.append("floc-faces")
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_names.append(roi)
            roi_classes.append("floc-places")
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_names.append(roi)
            roi_classes.append("floc-words")
        elif roi in [
            "early",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ]:
            roi_names.append(roi)
            roi_classes.append("streams")
        else:
            raise ValueError("Invalid ROI")
    return roi_names, roi_classes


def plot_interactive_surface(
    fmri,
    fs_indices,
    fs_coords,
    inverted_index_roi,
    min_x,
    max_x,
    min_y,
    max_y,
    roi_indices=None,
):
    locs = fs_coords[fs_indices]
    fs_coords = fs_coords[
        (fs_coords[:, 0] > min_x)
        & (fs_coords[:, 0] < max_x)
        & (fs_coords[:, 1] > min_y)
        & (fs_coords[:, 1] < max_y)
    ]
    ind = np.where(
        (locs[:, 0] > min_x)
        & (locs[:, 0] < max_x)
        & (locs[:, 1] > min_y)
        & (locs[:, 1] < max_y)
    )[0]
    if roi_indices is not None:
        ind = np.intersect1d(ind, roi_indices)
    rois = inverted_index_roi["rois"].values[fs_indices]
    data = fmri[ind]
    locs = locs[ind]
    rois = rois[ind]

    fig = go.Figure()

    fig.update_layout(
        hovermode="closest",
        width=1000,
        height=1000,
        showlegend=False,
        plot_bgcolor="#888888",
    )
    fig.update_xaxes(
        minallowed=min_x - 50,
        maxallowed=max_x + 50,
        visible=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        minallowed=min_y - 50,
        maxallowed=max_y + 50,
        visible=False,
        showticklabels=False,
    )
    fig.add_trace(
        go.Scatter(
            x=fs_coords[:, 0],
            y=fs_coords[:, 1],
            hoverinfo="skip",
            mode="markers",
            marker=dict(color="black", opacity=0.25),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=locs[:, 0],
            y=locs[:, 1],
            mode="markers",
            text=rois,
            customdata=fs_indices[ind],
            marker=dict(color=data, colorscale="RdBu_r", cmin=-3, cmax=3),
            hovertemplate="<b>%{text}</b><br><br>"
            + "FS Index: %{customdata}<br>"
            + "Activation: %{marker.color:.4f}<br>"
            + "<extra></extra>",
        )
    )

    fig.show()


def get_voxel_neighborhood(fs_indices, fs_coords, center_voxel, n_neighbor_voxels):
    assert center_voxel in fs_indices
    locs = fs_coords[fs_indices]
    center_coords = fs_coords[center_voxel]
    distances = np.linalg.norm(locs - center_coords, axis=1)
    sorted_indices = np.argsort(distances)
    neighborhood = sorted_indices[: n_neighbor_voxels + 1]
    return sorted(neighborhood.tolist())


def build_roi_inverted_index(data_root: str, hemisphere: int):
    inverted_index = {}
    for roi_class in tqdm(
        [
            "prf-visualrois",
            "floc-bodies",
            "floc-faces",
            "floc-places",
            "floc-words",
            "streams",
        ]
    ):
        # Load the ROI brain surface maps for the ROI class
        roi_class_dir = os.path.join(
            data_root,
            "roi_masks",
            hemisphere[0] + "h." + roi_class + "_fsaverage_space.npy",
        )
        roi_map_dir = os.path.join(
            data_root, "roi_masks", "mapping_" + roi_class + ".npy"
        )
        fsaverage_roi_class = np.load(roi_class_dir)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        # Iterate through corresponding ROIs
        roi_names, _ = parse_rois([roi_class])
        for roi in tqdm(roi_names):
            # Select the vertices corresponding to the ROI
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
            fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
            indices = np.where(fsaverage_roi == 1)[0]
            # Update the inverted index
            for idx in indices:
                inverted_index.setdefault(int(idx), [])
                inverted_index[int(idx)].append(roi)
    # Write in CSV
    data = []
    for i in range(len(fsaverage_roi)):
        data.append([sorted(inverted_index.get(i, []))])
    df = pd.DataFrame(data, columns=["rois"])
    df.to_csv(
        os.path.join(data_root, f"inverted_index_roi_{hemisphere}.csv"), index=False
    )
