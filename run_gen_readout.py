import os

import numpy as np
import torch
from PIL import Image

from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_measures import NSDMeasuresDataset
from methods.low_level_attributes.readout_guidance.readout_guidance import \
    rg_helpers


def get_control(config, control_range, image_dim):
    dataset = NSDMeasuresDataset(
    nsd= NaturalScenesDataset(
        root=config['dataset_root'],
        subject=config['subject'],
        partition="test",
    ),
    measures=config['measures'],
    patches_shape=config['patches_shape'],
    img_shape=config['img_shape'],
    )
    _, control = dataset[config['index']]
    control = control.to(config['device'])
    control_image = control.cpu().numpy()[0]
    control_image = np.clip(control_image, control_range[0], control_range[1])
    control_image = (control_image - control_range[0]) / (control_range[1] - control_range[0])
    control_image = (control_image * 255).astype(np.uint8)
    control_image = Image.fromarray(control_image).resize(image_dim)
    return control, control_image


def set_edits_control(
    config,
    edits, 
    image_dim,
):
    for edit in edits:
        if edit["head_type"] != "spatial":
            continue
        aggregation_config = edit["aggregation_kwargs"]
        control_range = aggregation_config["dataset_args"]["control_range"]
        control, control_image = get_control(config, control_range, image_dim)
        edit["control_image"] = control_image
        edit["control"] = control
        edit["control_range"] = control_range
    return edits, control_image


def main(config):

    # Load pipeline
    pipeline, dtype = rg_helpers.load_pipeline(config, config["device"])
    latent_height = latent_width = pipeline.unet.config.sample_size
    height = width = latent_height * pipeline.vae_scale_factor
    image_dim = (width, height)
    latent_dim = (latent_height, latent_width)

    # Create edit config and load aggregation network
    edits = rg_helpers.get_edits(config, config["device"], dtype)

    edits, control_image = set_edits_control(
        config,
        edits,
        image_dim,
    )

    prompts, latents = rg_helpers.get_prompts_latents(
        pipeline,
        config["prompt"],
        config["batch_size"], 
        config["seed"],
        latent_dim,
        config["device"],
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

    os.makedirs(config['output_dir'], exist_ok=True)
    control_image.resize(image_dim).save(os.path.join(config['output_dir'], 'control.png'))
    Image.fromarray(images[0]).resize(image_dim).save(os.path.join(config['output_dir'], 'image1.png'))
    Image.fromarray(images[1]).resize(image_dim).save(os.path.join(config['output_dir'], 'image2.png'))
    results[1].resize(image_dim).save(os.path.join(config['output_dir'], 'result1.png'))
    results[2].resize(image_dim).save(os.path.join(config['output_dir'], 'result2.png'))

if __name__ == "__main__":

    config = {

        'dataset_root': './data/NSD',
        'subject': 1,
        'index': 71,
        'measures': 'depth',
        'patches_shape': (64, 64),
        'img_shape': (448, 448),

        'model_path': 'runwayml/stable-diffusion-v1-5',
        'prompt': 'a photo of a kitchen',
        'output_dir': './data/readout_guidance/depth/examples',

        'batch_size': 2,
        'seed': 0,
        'device': torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),

        'rg_kwargs': [
            {
                'head_type': 'spatial',
                'loss_rescale': 1,
                'aggregation_kwargs': {'aggregation_ckpt': './data/readout_guidance/depth/checkpoints/last.pt'},
            }
        ],

        'generation_kwargs': {
            'text_weight': 7.5,
            'rg_weight': 2e-2,
            'rg_ratio': [0.0, 1.0],
            'eta': 1.0,
            'num_timesteps': 100,
            'negative_prompt': '',
        }
    }

    main(config)