import os
from PIL import Image
from torchvision import transforms
import numpy as np
from run_train_encoder import EncoderModule
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

for subject in range(1, 9):

    submission_dir = f'data/algonauts_2023_challenge_submission/subj0{subject}/'
    os.makedirs(submission_dir, exist_ok=True)

    ckpt_path_left = f'/gpfs/work5/0/gusr53691/data/checkpoints/0{subject}_all_l_all_0'
    ckpt_path_left = os.path.join(ckpt_path_left, sorted(list(os.listdir(ckpt_path_left)))[-1])

    ckpt_path_right = f'/gpfs/work5/0/gusr53691/data/checkpoints/0{subject}_all_r_all_0'
    ckpt_path_right = os.path.join(ckpt_path_right, sorted(list(os.listdir(ckpt_path_right)))[-1])
    
    img_folder = f'/gpfs/work5/0/gusr53691/data/NSD/subj0{subject}/test_split/test_images'
    img_list = os.listdir(img_folder)
    img_list = sorted([os.path.join(img_folder, img) for img in img_list])

    model = EncoderModule.load_from_checkpoint(ckpt_path_left).to(device).eval()
    
    preds = []
    for img in tqdm(img_list):
        img = Image.open(img).convert("RGB")
        img = transform(img).to(device)
        pred = model(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        preds.append(pred)

    preds = np.stack(preds, axis=0).astype(np.float32)
    np.save(os.path.join(submission_dir, 'lh_pred_test.npy'), preds)

    model = EncoderModule.load_from_checkpoint(ckpt_path_right).to(device).eval()

    preds = []
    for img in tqdm(img_list):
        img = Image.open(img).convert("RGB")
        img = transform(img).to(device)
        pred = model(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        preds.append(pred)

    preds = np.stack(preds, axis=0).astype(np.float32)
    np.save(os.path.join(submission_dir, 'rh_pred_test.npy'), preds)