import os

import numpy as np
import open_clip
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset


def load_modulation_vector(subject, roi, dataset_root, lmbda=10):

    subjects = [1,2,3,4,5,6,7,8]
    subjects.remove(subject)
    modulation_vectors = []

    for s in subjects:
        try:
            dataset_train_left = NSDCLIPFeaturesDataset(
                nsd=NaturalScenesDataset(
                    root=dataset_root,
                    subject=s,
                    partition="train",
                    hemisphere='left',
                    roi=roi,
                    return_average=True
                ),
                clip_extractor_type='clip_2_0'
            )
            dataset_train_right = NSDCLIPFeaturesDataset(
                nsd=NaturalScenesDataset(
                    root=dataset_root,
                    subject=s,
                    partition="train",
                    hemisphere='right',
                    roi=roi,
                    return_average=True
                ),
                clip_extractor_type='clip_2_0'
            )
        except:
            continue

        Y_train = (dataset_train_left.nsd.activations.numpy() + dataset_train_right.nsd.activations.numpy() ) / 2
        X_train = dataset_train_right.features / np.linalg.norm(dataset_train_right.features, axis=1, keepdims=True)
    
        W = X_train.T @ X_train + lmbda * np.eye(X_train.shape[1])
        W = (np.linalg.inv(W) @ X_train.T @ Y_train).T
        W_orig = W / np.linalg.norm(W)
        modulation_vectors.append(W_orig)
    
    modulation_vectors = np.stack(modulation_vectors, axis=0)
    modulation_vector = modulation_vectors.mean(axis=0)
    modulation_vector = modulation_vector / np.linalg.norm(modulation_vector)

    return modulation_vector

def order_by_modulation_vector(modulation_vector, features, return_sims=False):
    sims = 1 - pairwise_distances(modulation_vector.reshape(1,-1), features, metric="cosine")[0]
    order = np.argsort(sims)
    if return_sims:
        return order, sims
    return order

def save_for_attribute_pairs(
    attribute_pairs_path: str = "./data/attribute_pairs.tsv",
    save_dir: str = "./data/modulation_vectors/attribute_pairs",
):
    # Load the CLIP model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k").to(device)
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # Load the attribute pairs
    df = pd.read_csv(attribute_pairs_path, sep='\t')
    os.makedirs(save_dir, exist_ok=True)

    for description, text1, text2 in tqdm(df[['description', 'text1', 'text2']].values, total=len(df)):

        with torch.no_grad():
            # Encode the text using CLIP
            text = tokenizer([text1, text2]).to(device)
            x = clip.encode_text(text)
            x0 = x[0].float().detach().cpu().numpy()
            x1 = x[1].float().detach().cpu().numpy()
            # Shift vector is the difference between the two attribute vectors
            modulation_vector = x1 - x0
            modulation_vector /= np.linalg.norm(modulation_vector)
            np.save(os.path.join(save_dir, f'{description}.npy'), modulation_vector)

def save_for_nouns(
    nouns_path: str = "./data/laion_nouns.txt",
    save_dir: str = "./data/modulation_vectors/nouns",
):
    # Load the CLIP model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k").to(device)
    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # Load the nouns
    with open(nouns_path, 'r') as file:
        lines = file.readlines()
        nouns = [line.strip() for line in lines]
    os.makedirs(save_dir, exist_ok=True)

    for noun in tqdm(nouns, total=len(nouns)):

        with torch.no_grad():
            # Encode the text using CLIP
            text = tokenizer([f'a photo of {noun}']).to(device)
            modulation_vector = clip.encode_text(text)[0].float().detach().cpu().numpy()
            modulation_vector /= np.linalg.norm(modulation_vector)
            np.save(os.path.join(save_dir, f'{noun}.npy'), modulation_vector)