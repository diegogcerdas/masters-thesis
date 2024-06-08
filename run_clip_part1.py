import os, argparse
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm
from datasets.nsd.utils.nsd_utils import (get_roi_indices, parse_rois)
from methods.high_level_attributes.clip_extractor import create_clip_extractor


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} Hemisphere {cfg.hemisphere} ####')
    print('##############################')

    clip = create_clip_extractor('clip_2_0', cfg.device)

    roi_names, roi_classes = parse_rois([cfg.roi])
    roi_indices = get_roi_indices(os.path.join('/gpfs/work5/0/gusr53691/data/NSD/', f"subj{cfg.subject:02d}"), roi_names, roi_classes, cfg.hemisphere)

    if cfg.hemisphere == 'left':
        ckpt_path = os.path.join(cfg.ckpt_dir, 'clip_linear', f'0{cfg.subject}_all_l_all_0')
    else:
        ckpt_path = os.path.join(cfg.ckpt_dir, 'clip_linear', f'0{cfg.subject}_all_r_all_0')
    ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])

    encoder = EncoderModule.load_from_checkpoint(ckpt_path).to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}_{cfg.hemisphere}')
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
            x = clip(img)
            pred = encoder(x).squeeze(0).detach().cpu().numpy()
            feat = x.detach().cpu().numpy()
            preds.append(pred)
            features.append(feat)

        features = np.concatenate(features, axis=0).astype(np.float32)
        preds = np.stack(preds, axis=0).astype(np.float32)
        preds = preds[:,roi_indices].mean(-1)
        np.save(f1, features)
        np.save(f2, preds)
    
    print('##############################')
    print('### Finished ####')
    print('##############################')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--output_dir", type=str, default='.data/part1_outputs')
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="PPA")
    parser.add_argument("--hemisphere", type=str, default="right")
    
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