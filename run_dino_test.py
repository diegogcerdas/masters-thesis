import os, argparse
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm
from datasets.nsd.nsd import NaturalScenesDataset


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi}####')
    print('##############################')

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ckpt_path1 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi}_l_all_0')
    ckpt_path1 = os.path.join(ckpt_path1, sorted(list(os.listdir(ckpt_path1)))[-1])

    ckpt_path2 = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi}_r_all_0')
    ckpt_path2 = os.path.join(ckpt_path2, sorted(list(os.listdir(ckpt_path2)))[-1])

    model1 = EncoderModule.load_from_checkpoint(ckpt_path1, strict=False).to(cfg.device).eval()
    model2 = EncoderModule.load_from_checkpoint(ckpt_path2, strict=False).to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}')
    f1 = os.path.join(subj_folder, 'dino_vit_preds_left.npy')
    f2 = os.path.join(subj_folder, 'dino_vit_preds_right.npy')

    nsd = NaturalScenesDataset(
        root=cfg.dataset_root,
        subject=cfg.subject,
        partition='test',
    )

    preds1 = []
    preds2 = []

    for i in tqdm(range(len(nsd)), total=len(nsd)):

        img = nsd[i][0]
        img = transform_dino(img).to(cfg.device)
        pred1 = model1(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        preds1.append(pred1)
        pred2 = model2(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        preds2.append(pred2)

    preds1 = np.stack(preds1, axis=0).astype(np.float32)
    preds2 = np.stack(preds2, axis=0).astype(np.float32)
    np.save(f1, preds1)
    np.save(f2, preds2)


    print('##############################')
    print('### Finished ####')
    print('##############################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--output_dir", type=str, default='./data/dino_test')
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--roi", default="all")
    
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