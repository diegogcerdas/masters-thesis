import os, argparse
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm
from datasets.nsd.utils.nsd_utils import (get_roi_indices, parse_rois)
from methods.high_level_attributes.clip_extractor import create_clip_extractor
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset
from datasets.nsd.nsd import NaturalScenesDataset
from methods.high_level_attributes.shift_vectors import load_shift_vector


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} Hemisphere {cfg.hemisphere} Subset {cfg.subset} ####')
    print('##############################')

    clip = create_clip_extractor('clip_2_0', cfg.device)

    dataset = NSDCLIPFeaturesDataset(
        nsd = NaturalScenesDataset(
            root=cfg.dataset_root,
            subject=cfg.subject,
            partition='train',
            hemisphere=cfg.hemisphere,
            roi=cfg.roi,
            return_average=False,
            subset=cfg.subset
        ),
        clip_extractor_type='clip_2_0'
    )
    shift_vector = load_shift_vector(dataset)

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}_{cfg.hemisphere}_{cfg.subset}')
    subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        f1 = os.path.join(subj_folder, subfolder, 'clip_feats.npy')
        f2 = os.path.join(subj_folder, subfolder, 'clip_linear_preds.npy')

        if os.path.exists(f1) and os.path.exists(f2):
            continue

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        features = []
        preds = []
        for img in tqdm(img_list):
            img = Image.open(img).convert("RGB")
            img = transforms.ToTensor()(img).float()
            x = clip(img).squeeze(0).detach().cpu().numpy()
            pred = x @ shift_vector
            preds.append(pred)
            features.append(x)

        features = np.stack(features).astype(np.float32)
        preds = np.array(preds).astype(np.float32)
        np.save(f1, features)
        np.save(f2, preds)
    
    print('##############################')
    print('### Finished ####')
    print('##############################')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--output_dir", type=str, default='./data/part1_4_outputs')
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--roi", default="PPA")
    parser.add_argument("--hemisphere", type=str, default="right")
    parser.add_argument("--subset", type=str, default="wild_animals")
    
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # Parse arguments
    cfg = parser.parse_args()
    main(cfg)