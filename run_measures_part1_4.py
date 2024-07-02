import os, argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import visualpriors
from methods.low_level_attributes.xtc_network import UNet


def resize(measure):
    measure = torch.from_numpy(measure).float().unsqueeze(0)
    measure = torch.nn.functional.interpolate(
        measure,
        size=(64,64),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    measure = measure.numpy()
    return measure

def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} Hemisphere {cfg.hemisphere} Subset {cfg.subset} ####')
    print('##############################')

    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = zoe.to(cfg.device).eval()

    normal_model = UNet() 
    normals_path = os.path.join(cfg.data_root, 'xtc_checkpoints', 'rgb2normal_consistency.pth')
    normal_model.load_state_dict(torch.load(normals_path, map_location=cfg.device))
    normal_model = normal_model.to(cfg.device).eval()

    subj_folder = os.path.join(cfg.output_dir, f'{cfg.subject}_{cfg.roi}_{cfg.hemisphere}_{cfg.subset}')
    subfolders = sorted([f for f in os.listdir(subj_folder) if os.path.isdir(os.path.join(subj_folder, f))])

    for subfolder in tqdm(subfolders, total=len(subfolders)):

        img_list = np.array([f for f in os.listdir(os.path.join(subj_folder, subfolder)) if f.endswith('.png')])
        img_list_order = np.argsort([int(f.replace('.png', '')) for f in img_list])
        img_list = [os.path.join(subj_folder, subfolder, f) for f in img_list[img_list_order]]

        if os.path.exists(img_list[-1].replace('.png', '_gaussian_curvature.npy')):
            continue

        for img_f in tqdm(img_list):
            img = Image.open(img_f).convert("RGB")

            # compute depth
            img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
            with torch.no_grad():
                depth = zoe.infer(img_tensor).squeeze().detach().cpu().numpy()
                depth = depth[None, :, :]
                depth = resize(depth)
            f = img_f.replace('.png', '_depth.npy')
            np.save(f, depth)

            # compute surface normals
            img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
            with torch.no_grad():
                normal = normal_model(img_tensor)
                normal = torch.nn.functional.interpolate(
                    normal,
                    size=img.size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1).clamp(min=0, max=1)
                normal = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
                normal = np.moveaxis(normal, -1, 0)
                normal = resize(normal)
            f = img_f.replace('.png', '_surface_normal.npy')
            np.save(f, normal)

            # compute gaussian curvatures
            img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
            with torch.no_grad():
                principal_curvature = (visualpriors.feature_readout(img_tensor * 2 - 1, 'curvature', device=cfg.device) / 2. + 0.5)[:,:2]
                principal_curvature = torch.nn.functional.interpolate(
                    principal_curvature,
                    size=img.size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1).clamp(min=0, max=1)
                principal_curvature = principal_curvature.squeeze().permute(1,2,0).detach().cpu().numpy()
                gaussian_curvature = np.prod(principal_curvature, -1)[None,:,:]
                gaussian_curvature = resize(gaussian_curvature)
            f = img_f.replace('.png', '_gaussian_curvature.npy')
            np.save(f, gaussian_curvature)

    print('##############################')
    print('### Finished ####')
    print('##############################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--data_root", type=str, default="./data")
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