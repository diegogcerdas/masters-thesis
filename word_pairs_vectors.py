import pandas as pd
import open_clip
import torch
from tqdm import tqdm
import numpy as np
import os

df = pd.read_csv('word_pairs_wordnet.csv')

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

os.makedirs('word_pairs_vectors', exist_ok=True)

for i, (word1, word2) in tqdm(enumerate(df[['word1', 'word2']].values), total=len(df)):

    with torch.no_grad():

        text = tokenizer([word1, word2])
        x = clip.encode_text(text)
        shift_vector = (x[0] - x[1]).float().detach().cpu().numpy()
        shift_vector = shift_vector / np.linalg.norm(shift_vector)

        np.save(f'word_pairs_vectors/{i}.npy', shift_vector)