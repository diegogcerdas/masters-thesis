import pandas as pd
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

df = pd.read_csv('wordnet.csv')

candidate_labels = [
    'related to faces, eyes, nose, mouth',
    'related to hands, arms, fingers',
    'related to feet, legs, toes',
    'related to people, humans, persons',
    
    'related to animals, creatures, fauna',
    'related to plants, greenery, flora',

    'related to food, meals, eating',
    'related to furniture, household items',
    'related to tools, equipment, instruments',
    'related to clothing, textiles, garments',
    'related to electronics, gadgets, devices',
    'related to vehicles, transportation, travel',

    'related to written text, signs',

    'related to natural areas, landscapes, outdoors',
    'related to urban areas, buildings, structures',
    'related to indoors, rooms, interiors',
]

data = []

for id, desc in tqdm(df[['identifier','description']].values):
    results = classifier(desc, candidate_labels, multi_label=False)
    results = {k: v for k, v in zip(results['labels'], results['scores'])}
    results = [results[k] for k in candidate_labels]
    datapoint = [id] + results
    data.append(datapoint)

df = pd.DataFrame(data, columns=['identifier'] + candidate_labels)
df.to_csv('wordnet_scores.csv', index=False)