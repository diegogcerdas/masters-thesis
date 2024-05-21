import torch
from methods.low_level_attributes.readout_guidance.readout_guidance import rg_helpers


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


def main(config):

    # Load pipeline
    pipeline, dtype = rg_helpers.load_pipeline(config, config["device"])
    latent_height = latent_width = pipeline.unet.config.sample_size
    height = width = latent_height * pipeline.vae_scale_factor
    image_dim = (width, height)
    latent_dim = (latent_height, latent_width)

    # Create edit config and load aggregation network
    edits = rg_helpers.get_edits(config, config["device"], dtype)

    control_image = Image.open(control_path)
    edits = set_edits_control(
        edits,
        control_image, 
        image_dim, 
        latent_dim,
        config["device"]
    )
    control_image.resize(latent_dim[::-1])

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

if __name__ == "__main__":

    config = {

        'control_path': './depth.npy',

        'model_path': 'runwayml/stable-diffusion-v1-5',
        'num_timesteps': 100,
        'prompt': 'a photo of a train',
        'negative_prompt': '',

        'batch_size': 5,
        'seed': 0,
        'device': torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



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

    main(config)