import pandas as pd
import open_clip
import torch

df = pd.read_csv('adjective_pairs.csv')

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

vecs = []

for word1, word2 in df.values:

    text = tokenizer([word1, word2])
    x = clip.encode_text(text)
    shift_vector = (x[0] - x[1])
    vecs.append(shift_vector)

vecs = torch.stack(vecs)

torch.save(vecs, f'adjective_vectors.pt')