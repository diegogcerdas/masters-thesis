from model.feature_extractor import create_feature_extractor
import torch
from dataset.natural_scenes import NaturalScenesDataset
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
partition = 'debug_train'
data_root = './data/'

for feature_extractor_type in  ["alexnet_1", "alexnet_2", "alexnet_3", "alexnet_4", "alexnet_5", "alexnet_6"]:
    nsd = NaturalScenesDataset(data_root, subject, partition, return_coco_id=False)
    feature_extractor = create_feature_extractor(feature_extractor_type, device=device)
    f = os.path.join(data_root, f"subj{subject:02d}", "training_split", 'features', f'{feature_extractor_type}.npy')
    F = feature_extractor.extract_for_dataset(f, nsd, batch_size=1)
    print(f'{feature_extractor_type}: ', F.mean(), F.std())