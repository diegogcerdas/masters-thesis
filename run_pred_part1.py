import os
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm
from datasets.nsd.utils.nsd_utils import (get_roi_indices, parse_rois)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

subject = 1
hemisphere = 'right'
roi = 'PPA'

folder = '/gpfs/work5/0/gusr53691/data/part1_outputs'

for subject in range(1, 9):

    print(f'Processing subject {subject}...')

    roi_names, roi_classes = parse_rois([roi])
    roi_indices = get_roi_indices(os.path.join('/gpfs/work5/0/gusr53691/data/NSD/', f"subj{subject:02d}"), roi_names, roi_classes, hemisphere)

    if hemisphere == 'left':
        ckpt_path = f'/gpfs/work5/0/gusr53691/data/checkpoints/0{subject}_all_l_all_0'
    else:
        ckpt_path = f'/gpfs/work5/0/gusr53691/data/checkpoints/0{subject}_all_r_all_0'
    ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])

    model = EncoderModule.load_from_checkpoint(ckpt_path).to(device).eval()

    subj_folder = os.path.join(folder, f'{subject}_{roi}_{hemisphere}')
    subfolders = [f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))]

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        preds = []
        for img in tqdm(img_list):
            img = Image.open(img).convert("RGB")
            img = transform_dino(img).to(device)
            pred = model(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            preds.append(pred)

        preds = np.stack(preds, axis=0).astype(np.float32)
        preds = preds[:,roi_indices].mean(-1)
        np.save(os.path.join(subj_folder, subfolder, 'dino_vit_preds.npy'), preds)