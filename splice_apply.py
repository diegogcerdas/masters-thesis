import torch
import splice
import numpy as np
import pandas as pd

# parameters
l1_penalty = 0.0
concepts_dir = './attribute_vectors'
concepts = 'attributes.tsv'

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

vector = torch.from_numpy(np.load(f'ppa_vector.npy')).float().to(device).reshape(1,-1)

splicemodel = splice.load(concepts_dir, device, l1_penalty=l1_penalty)
weights, l0_norm, cosine = splice.decompose_vector(vector, splicemodel, device)

vocab = [desc for desc in pd.read_csv(concepts, sep='\t')['description']] #+ [f'{desc}_inv' for desc in pd.read_csv(concepts, sep='\t')['description']]

_, indices = torch.sort(weights, descending=True)

sum = 0

data = []
for idx in indices.squeeze():
    weight = weights[0, idx.item()].item()
    if weight == 0:
        break
    print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weight, 4)) + "\n")
    data.append((str(vocab[idx.item()]), weight))
    sum += weight

df = pd.DataFrame(data, columns=['description', 'weight'])
df.to_csv(f'ppa.csv', index=False)

print(l0_norm, cosine, sum)