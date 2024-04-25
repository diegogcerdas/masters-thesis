import torch
import splice
import numpy as np
import pandas as pd

# parameters
PC = 0

l1_penalty = 0.25
concepts_dir = './adjective_vectors'
concepts = 'adjective_pairs_wordnet.csv'

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

vector = torch.from_numpy(np.load(f'pc_vectors/{PC}.npy')).to(device).reshape(1,-1)
vocab = [f'{word1}_{word2}' for word1, word2 in pd.read_csv(concepts).values]

splicemodel = splice.load(concepts_dir, device, l1_penalty=l1_penalty)
weights, l0_norm, cosine = splice.decompose_vector(vector, splicemodel, device)

_, indices = torch.sort(weights, descending=True)

for idx in indices.squeeze():

    weight = weights[0, idx.item()].item()
    if weight == 0:
        break
    print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weight, 4)) + "\n")