import os, argparse
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm
from datasets.nsd.utils.nsd_utils import (get_roi_indices, parse_rois)


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} Hemisphere {cfg.hemisphere} ####')
    print('##############################')

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ckpt_path_left = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_all_l_all_0')
    ckpt_path_left = os.path.join(ckpt_path_left, sorted(list(os.listdir(ckpt_path_left)))[-1])
    ckpt_path_right = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_all_r_all_0')
    ckpt_path_right = os.path.join(ckpt_path_right, sorted(list(os.listdir(ckpt_path_right)))[-1])

    model_left = EncoderModule.load_from_checkpoint(ckpt_path_left, strict=False).to(cfg.device).eval()
    model_right = EncoderModule.load_from_checkpoint(ckpt_path_right, strict=False).to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}_{cfg.hemisphere}')
    subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        preds_left = []
        preds_right = []
        for img in tqdm(img_list):
            img = Image.open(img).convert("RGB")
            img = transform_dino(img).to(cfg.device)
            pred_l = model_left(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            pred_r = model_right(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            preds_left.append(pred_l)
            preds_right.append(pred_r)

        preds_left = np.stack(preds_left, axis=0).astype(np.float32)
        preds_right = np.stack(preds_right, axis=0).astype(np.float32)

        # left rois
        for roi in ['OFA', 'FFA-1', 'FFA-2', 'EBA', 'FBA-1', 'FBA-2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2']:
            try:
                roi_names, roi_classes = parse_rois([roi])
                roi_indices = get_roi_indices(os.path.join('/gpfs/work5/0/gusr53691/data/NSD/', f"subj{cfg.subject:02d}"), roi_names, roi_classes, 'left')
                preds = preds_left[:,roi_indices].mean(-1)
                f = os.path.join(subj_folder, subfolder, f'dino_vit_preds_{roi}_left.npy')
                np.save(f, preds)
            except:
                continue

        # right rois
        for roi in ['OFA', 'FFA-1', 'FFA-2', 'EBA', 'FBA-1', 'FBA-2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2']:
            try:
                roi_names, roi_classes = parse_rois([roi])
                roi_indices = get_roi_indices(os.path.join('/gpfs/work5/0/gusr53691/data/NSD/', f"subj{cfg.subject:02d}"), roi_names, roi_classes, 'right')
                preds = preds_right[:,roi_indices].mean(-1)
                f = os.path.join(subj_folder, subfolder, f'dino_vit_preds_{roi}_right.npy')
                np.save(f, preds)
            except:
                continue


    print('##############################')
    print('### Finished ####')
    print('##############################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--output_dir", type=str, default='./data/part1_outputs')
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