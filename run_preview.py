import argparse, os, json

import numpy as np
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms

from methods.ddim_inversion import ddim_inversion
from methods.img_utils import save_images
from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_clip import NSDCLIPFeaturesDataset

from methods.high_level_attributes.shift_vectors import load_shift_vector
from methods.high_level_attributes.clip_extractor import create_clip_extractor

from methods.low_level_attributes.image_measures import compute_brightness, compute_saturation, compute_warmth, compute_entropy

from run_train_encoder import EncoderModule
from methods.low_level_attributes.xtc_network import UNet
import visualpriors
from PIL import Image


def main(cfg):

    print('##############################')
    print(f'### Subject {cfg.subject} ####')
    print('##############################')

    # Since we use stabilityai/stable-diffusion-2-1-unclip, we fix the clip model to ViT-H-14 (laion2b_s32b_b79k)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-unclip"
    ddim_pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
    resolution = (768,768)

    source_img = Image.open(cfg.image_path).resize(resolution)
    # Perform DDIM inversion
    inverted_latents, _ = ddim_inversion(
        pretrained_model_name_or_path=ddim_pretrained_model_name_or_path,
        image=source_img,
        num_inference_steps=50,
        prompt=cfg.prompt,
        guidance_scale=1,
        seed=cfg.seed,
        device=cfg.device,
    )

    # Load unCLIP pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path).to(cfg.device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    pipe.safety_checker = None

    # Get CLIP vision embeddings
    source_img = pipe.feature_extractor(images=source_img, return_tensors="pt").pixel_values.to(device=cfg.device, dtype=dtype)
    source_img_embeds = pipe.image_encoder(source_img).image_embeds

    for roi in ['OFA', 'FFA-1', 'EBA', 'FBA-1', 'OWFA', 'VWFA-1', 'OPA', 'PPA', 'RSC']:

        # Folder for all results
        folder = os.path.join(cfg.output_dir, f"{cfg.subject}_{roi}")

        # Load shift vector from rest of subjects
        shift_vector_numpy = load_shift_vector(cfg.subject, roi, cfg.dataset_root)
        shift_vector = torch.from_numpy(shift_vector_numpy).to(cfg.device, dtype=dtype)

        # Get embeddings
        endpoint1 = shift_vector *  np.linalg.norm(source_img_embeds.squeeze().detach().cpu().numpy())
        endpoint2 = -shift_vector *  np.linalg.norm(source_img_embeds.squeeze().detach().cpu().numpy())
        embs1 = slerp(source_img_embeds, endpoint1, cfg.num_frames, t1=cfg.t1).half().to(cfg.device)
        embs2 = slerp(source_img_embeds, endpoint2, cfg.num_frames, t1=cfg.t1).half().to(cfg.device).flip(0)[:-1]
        embs = torch.cat([embs2, embs1])

        images = []
        for emb_i, emb in enumerate(embs):

            # Generate image
            img_pil = pipe(
                latents=inverted_latents,
                prompt=cfg.prompt,
                generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
                image_embeds=emb,
                noise_level=0,
            ).images[0]
            images.append(img_pil)

            names = [f'{j:04d}' for j in range(cfg.num_frames*2-1)]
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
    parser.add_argument("--image_path", type=str, default="img.png")
    parser.add_argument("--prompt", type=str, default="A large table filled with fruits and a vase of flowers")
    parser.add_argument("--output_dir", type=str, default='./data/preview')
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--t1", type=float, default=0.5)
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
