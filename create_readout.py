import einops
import glob
import json
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image
import sys
import torch
from tqdm import tqdm

from methods.low_level_attributes.readout_guidance.readout_guidance import rg_operators, rg_helpers

def image_to_array(source, source_range):
    source = np.array(source)
    source = einops.rearrange(source, 'w h c -> c w h')
    # Normalize source to [-1, 1]
    source = source.astype(np.float32) / 255.0
    source = rg_operators.renormalize(source, (0, 1), source_range)
    return source

def preprocess_control(source, resize_size, control_range):
    width, height = source.size
    crop_size = min(source.size)
    crop_x = np.random.randint(0, width - crop_size + 1)
    crop_y = np.random.randint(0, height - crop_size + 1)
    crop_resize_img = lambda img: img.convert("RGB").crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)).resize(resize_size)
    source = crop_resize_img(source)
    return torch.from_numpy(image_to_array(source, control_range))

def set_edits_control(
    edits, 
    control_image, 
    image_dim, 
    latent_dim,
    device
):
    for edit in edits:
        if edit["head_type"] != "spatial":
            continue
        aggregation_config = edit["aggregation_kwargs"]
        control_range = aggregation_config["dataset_args"]["control_range"]
        sparse_loss = aggregation_config["dataset_args"]["sparse_loss"]
        control = preprocess_control(control_image, latent_dim, control_range)
        control = control.to(device)
        control_image = control_image.resize(image_dim)
        edit["control_image"] = control_image
        edit["control"] = control
        edit["control_range"] = control_range
        edit["sparse_loss"] = sparse_loss
    return edits
    
def main(config_path, device="cuda"):
    config = OmegaConf.load(config_path)
    
    # Load pipeline
    pipeline, dtype = rg_helpers.load_pipeline(config, device)
    batch_size = config["batch_size"]
    latent_height = latent_width = pipeline.unet.config.sample_size
    height = width = latent_height * pipeline.vae_scale_factor
    image_dim = (width, height)
    latent_dim = (latent_height, latent_width)

    # Init seeds
    np.random.seed(config["seed"])

    # Create root save folder
    save_folder = config["output_dir"]
    os.makedirs(save_folder, exist_ok=True)

    # Create edit config and load aggregation network
    control_paths = glob.glob(f"{config['control_root']}/*")
    name_prompt = json.load(open(config["prompt_file"]))
    edits = rg_helpers.get_edits(config, device, dtype)

    for control_path in tqdm(control_paths):
        # Create name save folder
        name = os.path.basename(control_path).split('.')[0]
        prompt = name_prompt[name]
        save_folder = f"{config['output_dir']}/{name}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Create edits
        control_image = Image.open(control_path)
        control_image.save(f"{save_folder}/control.png")
        edits = set_edits_control(
            edits,
            control_image, 
            image_dim, 
            latent_dim,
            device
        )
        for j in range(num_frames):
            prompts, latents = rg_helpers.get_prompts_latents(
                pipeline,
                prompt,
                batch_size, 
                frame_seeds[j],
                latent_dim,
                device,
                dtype,
            )
            images, results = rg_helpers.run_preset_generation(
                pipeline, 
                prompts, 
                latents, 
                edits, 
                latent_dim=latent_dim,
                **config["generation_kwargs"]
            )
            # Save results
            save_name = lambda prefix: f"{save_folder}/{prefix}_{str(j).zfill(5)}.png"
            Image.fromarray(images[0]).save(save_name("orig"))
            Image.fromarray(images[1]).save(save_name("rg"))
            gt_feat, obs_feat = results[1], results[2]
            gt_feat.save(save_name("orig_readout"))
            obs_feat.save(save_name("rg_readout"))
       
if __name__ == "__main__":

    config = {

        'control_path': '',

        'model_path': 'runwayml/stable-diffusion-v1-5',
        'num_timesteps': 100,
        'prompt': '',
        'negative_prompt': '',

        'batch_size': 5,
        'seed': 0,


        rg_kwargs:
        - head_type: spatial
            loss_rescale: 0.5
            aggregation_kwargs:
            aggregation_ckpt: weights/readout_sdxl_spatial_${control_type}.pt
        generation_kwargs:
        text_weight: 7.5
        rg_weight: 2e-2
        rg_ratio: [0.0, 0.5]
        eta: 1.0
        output_dir: results/spatial/${control_type}
        control_root: data/spatial/${control_type}
    }



    # python3 script_spatial.py configs/spatial.yaml
    config_path = sys.argv[1]
    main(config_path)