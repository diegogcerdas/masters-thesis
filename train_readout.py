import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.utils import data
from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_measures import NSDMeasuresDataset
from methods.low_level_attributes.readout_guidance.train_helpers import load_models, load_optimizer, get_hyperfeats, prepare_batch, log_grid

def train(
    config,
    diffusion_extractor, 
    aggregation_network, 
    optimizer,
    train_dataloader, 
    val_dataloader,
):

    aggregation_network = aggregation_network.to(config["device"])

    for epoch in range(config["num_epochs"]):

        #################### VALIDATION ####################

        if epoch % config["validation_epochs"] == 0:
            
            with torch.no_grad():

                for j, batch in tqdm(enumerate(val_dataloader)):

                    imgs, target = prepare_batch(batch, config)
                    pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
                    grid = log_grid(imgs, target, pred)
                    grid.save(os.path.join(config["results_folder"], f"epoch-{epoch}_b-{j}.png"))

                    # TODO: save model

        ##################### TRAINING #####################

        for batch in tqdm(train_dataloader):
  
            imgs, target = prepare_batch(batch, config)
            pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = F.mse_loss(pred, target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def main(config):

    diffusion_extractor, aggregation_network = load_models(config)
    config["output_resolution"] = diffusion_extractor.output_resolution
    config["load_resolution"] = diffusion_extractor.load_resolution

    optimizer = load_optimizer(config, aggregation_network)

    # Training set
    nsd = NaturalScenesDataset(
        root=config['dataset_root'],
        subject=config['subject'],
        partition="train",
    )
    train_set = NSDMeasuresDataset(
        nsd=nsd,
        measures=config['measures'],
        patches_shape=config['patches_shape'],
        img_shape=config['img_shape'],
        predict_average=True,
        return_images=True,
    )
    train_dataloader = data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    # Validation set
    nsd = NaturalScenesDataset(
        root=config['dataset_root'],
        subject=config['subject'],
        partition="test",
    )
    val_set = NSDMeasuresDataset(
        nsd=nsd,
        measures=config['measures'],
        patches_shape=config['patches_shape'],
        img_shape=config['img_shape'],
        predict_average=True,
        return_images=True,
    )
    val_dataloader = data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    os.makedirs(config["results_folder"], exist_ok=True)

    train(
        diffusion_extractor, 
        aggregation_network, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        config['num_epochs'],
        config['validation_epochs'],
        config['device'],
    )

if __name__ == "__main__":
    
    config = {
        # General
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 0,
        "batch_size": 8,
        "num_workers": 18,
        "lr": 1e-4,
        "num_epochs": 15,
        "validation_epochs": 1,
        "results_folder": "./data/readout_guidance/results",

        # Diffusion Extractor
        "model_id": "runwayml/stable-diffusion-v1-5",
        'num_timesteps': 1000,
        "save_timestep": [0],
        "prompt": "",
        "negative_prompt": "",
        "diffusion_mode": "generation",

        # Aggregation Network
        "projection_dim": 384,
        "aggregation_kwargs": {
            'use_output_head': True,
            'output_head_channels': 1,
            'bottleneck_sequential': False,
        },

        # Dataset
        "dataset_root": "./data/NSD",
        "subject": 1,
        "measures": ["warmth"],
        "patches_shape": (25, 25),
        "img_shape": (425, 425),
    }

    main(config)