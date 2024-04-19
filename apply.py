from diffusers import StableUnCLIPImg2ImgPipeline, StableDiffusionPipeline, DDIMInverseScheduler
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
import json
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from methods.feature_extractor import create_feature_extractor
from torchvision import transforms
from utils.img_utils import save_images

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

subject = 1
data_root = './data/NSD'
seed = 0

hemisphere = 'right'
rois = [
    'PPA',
    'OPA',
    'RSC',
    'OFA',
    'EBA',
    'OWFA',
]
subsets = [
    'animal_not_bird_cat_dog_person_vehicle',
    'cat+dog_not_person',
    'food_not_animal_person',
    'person_sports_not_animal_food_vehicle',
    'vehicle_not_animal_person',
]

img_f = 'monalisa.png'
output_dir = './outputs/monalisa'
mults = range(-15, 16)
prompt = 'a painting of the mona lisa'

def get_encoder(roi):
    nsd = NaturalScenesDataset(
        root=data_root,
        subject=subject,
        partition="train",
        hemisphere=hemisphere,
        roi=roi,
    )
    feature_extractor_type = "clip_2_0"
    metric = 'cosine'
    n_neighbors = 0
    dataset = NSDFeaturesDataset(
        nsd=nsd,
        feature_extractor_type=feature_extractor_type,
        predict_average=True,
        metric=metric,
        n_neighbors=n_neighbors,
        seed=seed,
        device=device,
        keep_features=True,
    )
    encoder = LinearRegression().fit(dataset.features, dataset.targets)
    return encoder

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip").to(device)
dtype = next(pipe.image_encoder.parameters()).dtype

img_test = Image.open(img_f).convert('RGB').resize((768,768))

# Get CLIP vision embeddings
img_test = pipe.feature_extractor(images=img_test, return_tensors="pt").pixel_values
img_test = img_test.to(device=device, dtype=dtype)
img_test_embeds = pipe.image_encoder(img_test).image_embeds

feature_extractor = create_feature_extractor("clip_2_0", device)

for roi in rois:
    
    encoder = get_encoder(roi)
    
    for subset in subsets:

        vector_f = f'./subsets/{subject}_{roi}_{hemisphere}/{subset}/shift_vector.npy'
        shift_vector = torch.from_numpy(np.load(vector_f)).to(device, dtype=dtype)

        images = []
        acts = []
        for i, mult in enumerate(mults):
            generator = torch.Generator(device=device).manual_seed(seed)
            emb = img_test_embeds + mult * shift_vector
            img = pipe(
                prompt=prompt,
                generator=generator,
                image_embeds=emb,
                noise_level=0
            ).images[0]
            images.append(img)

            feats = feature_extractor(transforms.ToTensor()(img))
            pred = encoder.predict(feats)
            acts.append(pred)

        acts = np.array(acts)

        save_dir = f'{output_dir}/{subject}_{roi}_{hemisphere}/{subset}'
        save_images(images, save_dir)
        np.save(f'{save_dir}/acts.npy', acts)
