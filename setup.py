from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset
from datasets.nsd.nsd_measures import NSDMeasuresDataset
from datasets.nsd.utils.coco_utils import download_coco_annotation_file
from methods.low_level_attributes.image_measures import *
from methods.high_level_attributes.shift_vectors import save_for_attribute_pairs
from datasets.nsd.utils.coco_utils import save_segmentation_masks, build_coco_category_search
from datasets.nsd.utils.nsd_utils import build_roi_inverted_index

data_root = "./data/"
dataset_root = os.path.join(data_root, "NSD")

# 0. Download COCO files
download_coco_annotation_file(dataset_root)

for subject in [1,2,3,4,5,6,7,8]:

    print(f"Processing subject {subject}...")

    # 1. Create dataset (and save info CSV file)
    nsd = NaturalScenesDataset(dataset_root, subject, "all")

    # 2. Save segmentation masks
    save_segmentation_masks(dataset_root, subject)

    # 3. Build indices for images and ROIs
    build_coco_category_search(dataset_root, subject)
    build_roi_inverted_index(dataset_root, subject, 'left')
    build_roi_inverted_index(dataset_root, subject, 'right')

    # 4. Compute and save image measures
    save_depths(dataset_root, subject)
    save_surface_normals(dataset_root, subject)
    save_gaussian_curvatures(dataset_root, subject)

    # 5. Compute and save CLIP features
    _ = NSDCLIPFeaturesDataset(nsd, 'clip_1_5', predict_average=True)
    _ = NSDCLIPFeaturesDataset(nsd, 'clip_2_0', predict_average=True)

    # # 6. Compute measure average and stdev
    nsd = NaturalScenesDataset(dataset_root, subject, "train")
    measures = ["depth", "surface_normal", "gaussian_curvature", "warmth", "saturation", "brightness", "entropy"]
    _ = NSDMeasuresDataset(nsd, measures, (425, 425), (425, 425), predict_average=True)

# 7. Compute and save high-level attribute pair shift vectors
save_for_attribute_pairs(
    os.path.join(data_root, "attribute_pairs.tsv"),
    os.path.join(data_root, "shift_vectors/attribute_pairs")
)
