import pandas as pd
import open_clip
import torch
from tqdm import tqdm
import numpy as np
import os

df = pd.read_csv('attributes.tsv', sep='\t')

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

os.makedirs('attribute_vectors', exist_ok=True)

for i, (description1, description2) in tqdm(enumerate(df[['description1', 'description2']].values), total=len(df)):

    with torch.no_grad():

        text = tokenizer([description1, description2])
        x = clip.encode_text(text)
        x0 = x[0].float().detach().cpu().numpy()
        x1 = x[1].float().detach().cpu().numpy()

        np.save(f'attribute_vectors/{i}_0.npy', x0)
        np.save(f'attribute_vectors/{i}_1.npy', x1)