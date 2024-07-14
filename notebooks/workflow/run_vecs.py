import numpy as np
import open_clip
import os
from tqdm import tqdm
import torch
import pandas as pd

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
clip = clip.to(device)
tokenizer = open_clip.get_tokenizer("ViT-H-14")

f = 'wordnet.csv'
df = pd.read_csv(f)

save_dir = 'wordnet_vecs.npy'
word_vecs = np.empty((len(df), 1024))

for i, desc in tqdm(enumerate(df['name'].values), total=len(df)):
  temps = [template.format(desc) for template in templates]
  vectors = []
  for temp in temps:
    with torch.no_grad():
      text = tokenizer([temp]).to(device)
      shift_vector = clip.encode_text(text)[0].float().detach().cpu().numpy()
      vectors.append(shift_vector)
  shift_vector = np.mean(vectors, axis=0)
  word_vecs[i] = shift_vector

np.save(save_dir, word_vecs.astype(np.float32))