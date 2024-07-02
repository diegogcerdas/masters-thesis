import os, argparse
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} Hemisphere {cfg.hemisphere} Subset {cfg.subset} ####')
    print('##############################')

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    if cfg.hemisphere == 'left':
        ckpt_path= os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi}_l_all_0')
        ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])
    else:
        ckpt_path = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi}_r_all_0')
        ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])
    model = EncoderModule.load_from_checkpoint(ckpt_path, strict=False).to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}_{cfg.hemisphere}_{cfg.subset}')
    subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        f1 = os.path.join(subj_folder, subfolder, f'dino_vit_preds.npy')

        if os.path.exists(f1):
            continue

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        preds = []
        for img in tqdm(img_list):
            img = Image.open(img).convert("RGB")
            img = transform_dino(img).to(cfg.device)
            pred = model(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            preds.append(pred)

        preds = np.stack(preds, axis=0).astype(np.float32).mean(-1)

        np.save(f1, preds)


    print('##############################')
    print('### Finished ####')
    print('##############################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--output_dir", type=str, default='./data/part1_4_outputs')
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