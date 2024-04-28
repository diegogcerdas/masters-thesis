import pandas as pd
import open_clip
import torch
from tqdm import tqdm
import numpy as np
import os

with open('categories.txt', 'r') as file:
    lines = file.readlines()

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

os.makedirs('category_vectors', exist_ok=True)

for l in tqdm(lines, total=len(lines)):

    l = l.strip()

    with torch.no_grad():

        text = tokenizer([l])
        x = clip.encode_text(text)
        x = x[0].float().detach().cpu().numpy()

        np.save(f'category_vectors/{l}.npy', )