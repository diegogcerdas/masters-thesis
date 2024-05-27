import numpy as np
import pandas as pd
import open_clip
from tqdm import tqdm
import torch
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances

def compute_for_activations(targets, features):
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]
    model = LinearRegression(fit_intercept=False).fit(targets, features - features.mean(axis=0))
    shift_vectors = model.coef_ / np.linalg.norm(model.coef_, axis=0)
    return shift_vectors

def order_by_shift_vector(shift_vector, features, return_sims=False):
    sims = 1 - pairwise_distances(shift_vector.reshape(1,-1), features, metric="cosine")[0]
    order = np.argsort(sims)
    if return_sims:
        return order, sims
    return order

def save_for_attribute_pairs(
    attribute_pairs_path: str = "./data/attribute_pairs.tsv",
    save_dir: str = "./data/shift_vectors/attribute_pairs",
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
            shift_vector = x1 - x0
            shift_vector /= np.linalg.norm(shift_vector)
            np.save(os.path.join(save_dir, f'{description}.npy'), shift_vector)

def save_for_nouns(
    nouns_path: str = "./data/laion_nouns.txt",
    save_dir: str = "./data/shift_vectors/nouns",
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
            shift_vector = clip.encode_text(text)[0].float().detach().cpu().numpy()
            shift_vector /= np.linalg.norm(shift_vector)
            np.save(os.path.join(save_dir, f'{noun}.npy'), shift_vector)