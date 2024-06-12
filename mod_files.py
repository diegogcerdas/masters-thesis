import os
from tqdm import tqdm
import numpy as np
from methods.img_utils import save_images
from PIL import Image

output_dir = 'data/part1_outputs'

hemisphere = 'right'
subjects = [2,3,4,5,6,7,8]
rois = ['OFA', 'FFA-1', 'FFA-2', 'EBA', 'FBA-1', 'FBA-2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2']

for subject in tqdm(subjects):
    
    for roi in tqdm(rois):

        subj_folder = os.path.join(output_dir, f'{subject}_{roi}_{hemisphere}')
        if not os.path.exists(subj_folder):
            continue
        subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

        for subfolder in subfolders:

            img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
            img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
            img_list = np.array([os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]])

            if len(img_list) == 11:
                continue

            img_list = img_list[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]
            img_list = [Image.open(f) for f in img_list]

            clip_feats = np.load(os.path.join(subj_folder, subfolder, 'clip_feats.npy'))
            clip_feats = clip_feats[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]

            clip_linear_preds = np.load(os.path.join(subj_folder, subfolder, 'clip_linear_preds.npy'))
            clip_linear_preds = clip_linear_preds[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]
            dino_vit_preds = np.load(os.path.join(subj_folder, subfolder, 'dino_vit_preds.npy'))
            dino_vit_preds = dino_vit_preds[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]

            # delete original folder
            os.system(f'rm -rf {os.path.join(subj_folder, subfolder)}')

            names = [f'{j:04d}' for j in range(len(img_list))]
            save_images(img_list, os.path.join(subj_folder, subfolder), names)
            np.save(os.path.join(subj_folder, subfolder, 'clip_feats.npy'), clip_feats)
            np.save(os.path.join(subj_folder, subfolder, 'clip_linear_preds.npy'), clip_linear_preds)
            np.save(os.path.join(subj_folder, subfolder, 'dino_vit_preds.npy'), dino_vit_preds)