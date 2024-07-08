import argparse, os, json

import numpy as np
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

from methods.ddim_inversion import ddim_inversion
from methods.img_utils import save_images
from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset

from methods.high_level_attributes.shift_vectors import load_shift_vector


def main(cfg):

    print('##############################')
    print(f'### Category {cfg.category} ####')
    print('##############################')

    # Since we use stabilityai/stable-diffusion-2-1-unclip, we fix the clip model to ViT-H-14 (laion2b_s32b_b79k)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-unclip"
    ddim_pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
    resolution = (768,768)

    # Load unCLIP pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path).to(cfg.device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    pipe.safety_checker = None  

    # Load validation dataset
    nsd = NaturalScenesDataset(
        root=cfg.dataset_root,
        subject=1,
        partition="test",
    )
    f = os.path.join(nsd.subj_dir, 'nsd_idx2captions.json')
    with open(f, 'r') as file:
        nsd_idx2captions = json.load(file)

    # Select subset
    rng = np.random.default_rng(cfg.seed)
    subset = rng.choice(range(len(nsd)), len(nsd), replace=False)[:cfg.num_images]

    # Load shift vector
    shift_vector = np.load(os.path.join(cfg.vectors_dir, f'{cfg.category}.npy'))
    shift_vector = torch.from_numpy(shift_vector).to(cfg.device, dtype=dtype)

    for i in sorted(subset):

        folder = os.path.join(cfg.output_dir, f"{cfg.category}/{i:04d}")
        if os.path.exists(folder):
            continue

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
        endpoint = shift_vector *  np.linalg.norm(source_img_embeds.squeeze().detach().cpu().numpy())
        embs = slerp(source_img_embeds, endpoint, cfg.num_frames, t1=cfg.t1).half().to(cfg.device)

        images = []
        for emb in embs:

            # Generate image
            img = pipe(
                latents=inverted_latents,
                prompt=prompt,
                generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
                image_embeds=emb,
                noise_level=0,
            ).images[0]
            images.append(img)

        names = [f'{j:04d}' for j in range(cfg.num_frames)]
        save_images(images, folder, names)

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
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--vectors_dir", type=str, default="./data/shift_vectors/categories")
    parser.add_argument("--category", type=str, default="animals")

    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--t1", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default='./data/part2_outputs')
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
