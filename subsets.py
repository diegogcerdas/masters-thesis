import torch
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
import json
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
data_root = './data/NSD'
seed = 0

# Load index and inverted index
f = os.path.join(data_root, f'subj{subject:02d}/index_train.json')
index = json.load(open(f, 'r'))
f = os.path.join(data_root, f'subj{subject:02d}/inverted_index_train.json')
inv_index = json.load(open(f, 'r'))

for hemisphere in ['right']: 
    for roi in ['PPA', 'RSC', 'OPA', 'OFA', 'FFA-1']:  # ['PPA', 'RSC', 'OPA', 'OFA', 'FFA-1', 'FFA-2', 'EBA', 'OWFA']:

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

        encoder = LinearRegression().fit(dataset.features, dataset.targets)
        y_pred = encoder.predict(dataset.features)
        metric = encoder.score(dataset.features, dataset.targets)

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

        for subset in ['wild_animals', 'people_sports', 'pets', 'vehicles', 'food']:

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
            remaining_negatives = [(k,v) for k, v in sorted(remaining_negatives.items(), key=lambda item: item[1], reverse=True)]

            targets = y_pred
            targets = targets[all_indices]
            num_inputs = 50

            # Select Max-EIs
            max_eis_idx_sample = np.argsort(targets)[-num_inputs:]
            max_eis_act = targets[max_eis_idx_sample].mean()
            max_eis_idx_sample = all_indices[max_eis_idx_sample]
            max_eis_images = [Image.open(os.path.join(data_root, dataset.nsd.df.iloc[max_eis_idx_sample[i]]["filename"])) for i in range(num_inputs)]

            # Select Min-EIs
            min_eis_idx_sample = np.argsort(targets)[:num_inputs]
            min_eis_act = targets[min_eis_idx_sample].mean()
            min_eis_idx_sample = all_indices[min_eis_idx_sample]
            min_eis_images = [Image.open(os.path.join(data_root, dataset.nsd.df.iloc[min_eis_idx_sample[i]]["filename"])) for i in range(num_inputs)]

            pos_str = '+'.join(sorted(['_'.join(sorted(p)) for p in all_positives]))
            neg_str = '_'.join(sorted(negatives))
            query = f'{pos_str}_not_{neg_str}'

            folder = f'./subsets/{subject}_{roi}_{hemisphere}/{query}/max'
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, 'indices.npy'), max_eis_idx_sample)
            with open(os.path.join(folder, 'activation.txt'), 'w') as f:
                f.write(str(np.round(max_eis_act, 4)))

            folder = f'./subsets/{subject}_{roi}_{hemisphere}/{query}/min'
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, 'indices.npy'), min_eis_idx_sample)
            with open(os.path.join(folder, 'activation.txt'), 'w') as f:
                f.write(str(np.round(min_eis_act, 4)))