import torch
import splice
import numpy as np
import pandas as pd

# parameters
PC = 0

l1_penalty = 0.1
concepts_dir = './word_pairs_vectors'
concepts = 'word_pairs_wordnet.csv'

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

vector = torch.from_numpy(np.load(f'pc_vectors/{PC}.npy')).float().to(device).reshape(1,-1)
indices = [i for i, (_, _, pos) in enumerate(pd.read_csv(concepts).values) if pos in ['a', 's', 'r', 'v']]

splicemodel, indices = splice.load(concepts_dir, indices, device, l1_penalty=l1_penalty)
weights, l0_norm, cosine = splice.decompose_vector(vector, splicemodel, device)

vocab = [f'{word1}_{word2}' for word1, word2, _ in pd.read_csv(concepts).values[indices]]

_, indices = torch.sort(weights, descending=True)

data = []
for idx in indices.squeeze():
    weight = weights[0, idx.item()].item()
    if weight == 0:
        break
    print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weight, 4)) + "\n")
    data.append((str(vocab[idx.item()]), weight))

df = pd.DataFrame(data, columns=['pair', 'weight'])
df.to_csv(f'pc{PC}.csv', index=False)

print(l0_norm, cosine)