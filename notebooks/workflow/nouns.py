from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np

zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")

hypothesis_template = "This word can be categorized as {}"
candidate_labels=["human", "animal", "plant", "vehicle", "food", "clothing", "body part", "furniture", "electronics", "item, tool or appliance", "manmade area", "natural area", "event", "feeling or emotion"]


with open('laion_nouns.txt', 'r') as file:
    lines = file.readlines()
    nouns = [line.strip() for line in lines]

f = 'laion_nouns_classification.csv'

data = []

for noun in tqdm(nouns):
    result = zeroshot_classifier(noun, candidate_labels, hypothesis_template=hypothesis_template)
    data_n = [noun]
    for label in candidate_labels:
        idx = np.where(np.array(result['labels']) == label)[0][0]
        data_n.append(f"{result['scores'][idx]:.3f}")
    data.append(data_n)

data = pd.DataFrame(data, columns=['noun'] + candidate_labels)
data.to_csv(f, index=False)