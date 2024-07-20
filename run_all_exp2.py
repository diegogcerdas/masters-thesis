import argparse, os, json

import numpy as np
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms

from methods.ddim_inversion import ddim_inversion
from methods.img_utils import save_images
from datasets.nsd.nsd import NaturalScenesDataset

from methods.high_level_attributes.shift_vectors import load_shift_vector
from methods.high_level_attributes.clip_extractor import create_clip_extractor

from methods.low_level_attributes.image_measures import compute_brightness, compute_saturation, compute_warmth, compute_entropy

from run_train_encoder import EncoderModule
from methods.low_level_attributes.xtc_network import UNet
import visualpriors

def resize(measure):
    measure = torch.from_numpy(measure).float().unsqueeze(0)
    measure = torch.nn.functional.interpolate(
        measure,
        size=(10,10),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    measure = measure.numpy()
    return measure


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ROI1 {cfg.roi1} ROI2 {cfg.roi2} ####')
    print('##############################')

    measurements = \
    [f'brightness_{i+1}' for i in range(100)] + \
    [f'saturation_{i+1}' for i in range(100)] + \
    [f'warmth_{i+1}' for i in range(100)] + \
    [f'entropy_{i+1}' for i in range(100)] + \
    [f'depth_{i+1}' for i in range(100)] + \
    [f'gaussian_curvature_{i+1}' for i in range(100)] + \
    [f'surface_normal_1_{i+1}' for i in range(100)] + \
    [f'surface_normal_2_{i+1}' for i in range(100)] + \
    [f'surface_normal_3_{i+1}' for i in range(100)]

    subsets = ['wild_animals', 'birds', 'vehicles', 'food', 'sports', 'furniture']

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Measurement models
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = zoe.to(cfg.device).eval()

    normal_model = UNet() 
    normals_path = os.path.join(cfg.data_root, 'xtc_checkpoints', 'rgb2normal_consistency.pth')
    normal_model.load_state_dict(torch.load(normals_path, map_location=cfg.device))
    normal_model = normal_model.to(cfg.device).eval()

    # Folder for all results
    folder_all = os.path.join(cfg.output_dir, f"{cfg.subject}_{cfg.roi1}_{cfg.roi2}")

    # Initialize data arrays
    clip_feats = np.empty((len(subsets), cfg.num_images, cfg.num_frames*2-1, 1024), dtype=np.float32)
    dino_preds = np.empty((2, len(subsets), cfg.num_images, cfg.num_frames*2-1), dtype=np.float32)
    measures = np.empty((len(subsets), cfg.num_images, cfg.num_frames*2-1, len(measurements)), dtype=np.float32)

    # Since we use stabilityai/stable-diffusion-2-1-unclip, we fix the clip model to ViT-H-14 (laion2b_s32b_b79k)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-unclip"
    ddim_pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
    resolution = (768,768)
    clip = create_clip_extractor('clip_2_0', cfg.device)

    # Load unCLIP pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path).to(cfg.device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    pipe.safety_checker = None  

    # Load DINO-ViT models
    ckpt_path_left = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi1}_l_all_0')
    ckpt_path_left = os.path.join(ckpt_path_left, sorted(list(os.listdir(ckpt_path_left)))[-1])
    model_left_1 = EncoderModule.load_from_checkpoint(ckpt_path_left, strict=False).to(cfg.device).eval()

    ckpt_path_right = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi1}_r_all_0')
    ckpt_path_right = os.path.join(ckpt_path_right, sorted(list(os.listdir(ckpt_path_right)))[-1])
    model_right_1 = EncoderModule.load_from_checkpoint(ckpt_path_right, strict=False).to(cfg.device).eval()
    
    ckpt_path_left = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi2}_l_all_0')
    ckpt_path_left = os.path.join(ckpt_path_left, sorted(list(os.listdir(ckpt_path_left)))[-1])
    model_left_2 = EncoderModule.load_from_checkpoint(ckpt_path_left, strict=False).to(cfg.device).eval()

    ckpt_path_right = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi2}_r_all_0')
    ckpt_path_right = os.path.join(ckpt_path_right, sorted(list(os.listdir(ckpt_path_right)))[-1])
    model_right_2 = EncoderModule.load_from_checkpoint(ckpt_path_right, strict=False).to(cfg.device).eval()

    f = os.path.join(os.path.join(cfg.dataset_root, f"subj{cfg.subject:02d}"), 'nsd_idx2captions.json')
    with open(f, 'r') as file:
        nsd_idx2captions = json.load(file)

    # Load shift vector from rest of subjects
    # ROI 1
    shift_vector1 = load_shift_vector(cfg.subject, cfg.roi1, cfg.dataset_root)
    # ROI 2
    shift_vector2 = load_shift_vector(cfg.subject, cfg.roi2, cfg.dataset_root)

    shift_vector = shift_vector1 - shift_vector2
    shift_vector_numpy = shift_vector / np.linalg.norm(shift_vector)
    shift_vector = torch.from_numpy(shift_vector_numpy).to(cfg.device, dtype=dtype)

    for sub_i, subset in enumerate(['wild_animals', 'birds', 'vehicles', 'food', 'sports', 'furniture']):

        # Load validation dataset
        nsd = NaturalScenesDataset(
            root=cfg.dataset_root,
            subject=cfg.subject,
            partition="test",
            subset=subset,
        )
        f = os.path.join(nsd.subj_dir, 'nsd_idx2captions.json')
        with open(f, 'r') as file:
            nsd_idx2captions = json.load(file)

        # Select subset
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(range(len(nsd)), len(nsd), replace=False)[:cfg.num_images]

        for idx_i, i in enumerate(sorted(indices)):

            folder = os.path.join(folder_all, f"{subset}_{i:04d}")

            # Load source image and perform DDIM inversion
            source_img, nsd_idx = nsd[i]
            source_img = source_img.resize(resolution)
            prompt = nsd_idx2captions[str(nsd_idx)]
            inverted_latents, _ = ddim_inversion(
                pretrained_model_name_or_path=ddim_pretrained_model_name_or_path,
                image=source_img,
                num_inference_steps=50,
                prompt=prompt,
                guidance_scale=1,
                seed=cfg.seed,
                device=cfg.device,
            )
            
            # Get CLIP vision embeddings
            source_img = pipe.feature_extractor(images=source_img, return_tensors="pt").pixel_values.to(device=cfg.device, dtype=dtype)
            source_img_embeds = pipe.image_encoder(source_img).image_embeds

            # Get embeddings
            endpoint1 = shift_vector *  np.linalg.norm(source_img_embeds.squeeze().detach().cpu().numpy())
            endpoint2 = -shift_vector *  np.linalg.norm(source_img_embeds.squeeze().detach().cpu().numpy())
            embs1 = slerp(source_img_embeds, endpoint1, cfg.num_frames, t1=cfg.t1).half().to(cfg.device)
            embs2 = slerp(source_img_embeds, endpoint2, cfg.num_frames, t1=cfg.t1).half().to(cfg.device).flip(0)[:-1]
            embs = torch.cat([embs2, embs1])

            pipe_seed = np.random.randint(0, 1e+9)

            images = []
            for emb_i, emb in enumerate(embs):

                # Generate image
                img_pil = pipe(
                    latents=inverted_latents,
                    prompt=prompt,
                    generator=torch.Generator(device=cfg.device).manual_seed(pipe_seed),
                    image_embeds=emb,
                    noise_level=0,
                ).images[0]
                images.append(img_pil)

                img = transforms.ToTensor()(img_pil).float()
                feats = clip(img).squeeze(0).detach().cpu().numpy()
                pred = feats @ shift_vector_numpy
                clip_feats[sub_i, idx_i, emb_i] = feats

                img = transform_dino(img_pil).to(cfg.device)

                pred_l = model_left_1(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                pred_r = model_right_1(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                dino_preds[0, sub_i, idx_i, emb_i] = (pred_l + pred_r) / 2

                pred_l = model_left_2(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                pred_r = model_right_2(img.unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                dino_preds[1, sub_i, idx_i, emb_i] = (pred_l + pred_r) / 2

                ms = []

                # brightness
                m = compute_brightness(img_pil)
                m = resize(m)[0].reshape(-1).tolist()
                ms.extend(m)

                # saturation
                m = compute_saturation(img_pil)
                m = resize(m)[0].reshape(-1).tolist()
                ms.extend(m)

                # warmth
                m = compute_warmth(img_pil)
                m = resize(m)[0].reshape(-1).tolist()
                ms.extend(m)

                # entropy
                m = compute_entropy(img_pil)
                m = resize(m)[0].reshape(-1).tolist()
                ms.extend(m)

                # compute depth
                img_tensor = torch.tensor(np.array(img_pil).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
                with torch.no_grad():
                    depth = zoe.infer(img_tensor).squeeze().detach().cpu().numpy()
                    depth = depth[None, :, :]
                    depth = resize(depth)[0].reshape(-1).tolist()
                ms.extend(depth)

                # compute gaussian curvatures
                img_tensor = torch.tensor(np.array(img_pil.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
                with torch.no_grad():
                    principal_curvature = (visualpriors.feature_readout(img_tensor * 2 - 1, 'curvature', device=cfg.device) / 2. + 0.5)[:,:2]
                    principal_curvature = torch.nn.functional.interpolate(
                        principal_curvature,
                        size=img_pil.size,
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(1).clamp(min=0, max=1)
                    principal_curvature = principal_curvature.squeeze().permute(1,2,0).detach().cpu().numpy()
                    gaussian_curvature = np.prod(principal_curvature, -1)[None,:,:]
                    gaussian_curvature = resize(gaussian_curvature)[0].reshape(-1).tolist()
                ms.extend(gaussian_curvature)

                # compute surface normals
                img_tensor = torch.tensor(np.array(img_pil.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device) / 255
                with torch.no_grad():
                    normal = normal_model(img_tensor)
                    normal = torch.nn.functional.interpolate(
                        normal,
                        size=img_pil.size,
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(1).clamp(min=0, max=1)
                    normal = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
                    normal = np.moveaxis(normal, -1, 0)
                    normal = resize(normal)
                m_ = normal[0].reshape(-1).tolist()
                ms.extend(m_)
                m_ = normal[1].reshape(-1).tolist()
                ms.extend(m_)
                m_ = normal[2].reshape(-1).tolist()
                ms.extend(m_)

                measures[sub_i, idx_i, emb_i] = ms

            names = [f'{j:04d}' for j in range(cfg.num_frames*2-1)]
            save_images(images, folder, names)

    np.save(os.path.join(folder_all, 'clip_feats.npy'), clip_feats.astype(np.float32))
    np.save(os.path.join(folder_all, 'dino_preds.npy'), dino_preds.astype(np.float32))
    np.save(os.path.join(folder_all, 'measures.npy'), measures.astype(np.float32))

    print('##############################')
    print('### Finished ####')
    print('##############################')

def slerp(v0, v1, num, t0=0, t1=1):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2
    t = np.linspace(t0, t1, num)
    v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))
    return v3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--ckpt_dir", type=str, default="./data/checkpoints")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi1", default="OPA")
    parser.add_argument("--roi2", default="PPA")

    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--num_images", type=int, default=15)
    parser.add_argument("--t1", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default='./data/exp2_outputs')
    parser.add_argument("--seed", type=int, default=0)
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
