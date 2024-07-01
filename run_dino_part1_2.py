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
    print(f'### Subject {cfg.subject} ROI1 {cfg.roi1} ROI2 {cfg.roi2} Hemisphere {cfg.hemisphere} ####')
    print('##############################')

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if cfg.hemisphere == 'left':
        ckpt_path1 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi1}_l_all_0')
        ckpt_path1 = os.path.join(ckpt_path1, sorted(list(os.listdir(ckpt_path1)))[-1])
        ckpt_path2 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi2}_l_all_0')
        ckpt_path2 = os.path.join(ckpt_path2, sorted(list(os.listdir(ckpt_path2)))[-1])
    else:
        ckpt_path1 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi1}_r_all_0')
        ckpt_path1 = os.path.join(ckpt_path1, sorted(list(os.listdir(ckpt_path1)))[-1])
        ckpt_path2 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi2}_r_all_0')
        ckpt_path2 = os.path.join(ckpt_path2, sorted(list(os.listdir(ckpt_path2)))[-1])

    model1 = EncoderModule.load_from_checkpoint(ckpt_path1, strict=False).to(cfg.device).eval()
    model2 = EncoderModule.load_from_checkpoint(ckpt_path2, strict=False).to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi1}_{cfg.roi2}_{cfg.hemisphere}')
    subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        f1 = os.path.join(subj_folder, subfolder, 'dino_vit_preds1.npy')
        f2 = os.path.join(subj_folder, subfolder, 'dino_vit_preds2.npy')

        if os.path.exists(f1) and os.path.exists(f2):
            continue

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        preds1 = []
        preds2 = []
        for img in tqdm(img_list):
            img = Image.open(img).convert("RGB")
            img = transform_dino(img).to(cfg.device)
            pred1 = model1(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            preds1.append(pred1)
            pred2 = model2(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            preds2.append(pred2)

        preds1 = np.stack(preds1, axis=0).astype(np.float32).mean(-1)
        preds2 = np.stack(preds2, axis=0).astype(np.float32).mean(-1)
        np.save(f1, preds1)
        np.save(f2, preds2)


    print('##############################')
    print('### Finished ####')
    print('##############################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--output_dir", type=str, default='./data/part1_2_outputs')
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--roi1", default="OPA")
    parser.add_argument("--roi2", default="PPA")
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