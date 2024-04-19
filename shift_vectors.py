import torch
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
data_root = './data/NSD'
seed = 0
hemisphere = 'left'
roi = 'PPA'

for hemisphere in ['left', 'right']: 
    for roi in ['PPA', 'RSC', 'OPA', 'OFA', 'FFA-1', 'FFA-2', 'EBA', 'OWFA']:
        
        try:
            nsd = NaturalScenesDataset(
                root=data_root,
                subject=subject,
                partition="train",
                hemisphere=hemisphere,
                roi=roi,
            )
        except:
            continue

        feature_extractor_type = "clip_2_0"
        metric = 'cosine'
        n_neighbors = 0

        dataset = NSDFeaturesDataset(
        nsd=nsd,
        feature_extractor_type=feature_extractor_type,
        predict_average=True,
        metric=metric,
        n_neighbors=n_neighbors,
        seed=seed,
        device=device,
        keep_features=True,
        )
        del nsd

        subsets = [
            'animal_not_bird_cat_dog_person_vehicle',
            'cat+dog_not_person',
            'food_not_animal_person',
            'person_sports_not_animal_food_vehicle',
            'vehicle_not_animal_person',
        ]

        for subset in subsets:
        
            f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/max/indices.npy'
            max_indices = np.load(f)
            f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/min/indices.npy'
            min_indices = np.load(f)

            f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/max/activation.txt'
            with open(f, 'r') as file:
                max_act = float(file.readlines()[0])
            f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/min/activation.txt'
            with open(f, 'r') as file:
                min_act = float(file.readlines()[0])

            effect = max_act - min_act

            feats_max = dataset.features[max_indices]
            feats_min = dataset.features[min_indices]

            shift_vector = feats_max.mean(axis=0) - feats_min.mean(axis=0)
            shift_vector = shift_vector / (effect * 10)

            f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/shift_vector.npy'
            np.save(f, shift_vector)


