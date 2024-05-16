import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils import data
from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_measures import NSDMeasuresDataset
from readout_guidance.train_helpers import load_models, load_optimizer, get_hyperfeats

# ====================
#     Dataloader
# ====================
def get_spatial_loader(config, annotation_file, shuffle):
    dataset = ControlDataset(
        annotation_file,
        size=config["load_resolution"],
        **config["dataset_args"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    return dataset, dataloader


# ====================
#        Loss
# ====================
def loss_spatial(pred, target):
    target = train_helpers.standardize_feats(pred, target)
    loss = torch.nn.functional.mse_loss(pred, target)
    return loss

# ====================
#  Validate and Train
# ====================
def validate(diffusion_extractor, aggregation_network, dataloader, split):
    device = aggregation_network.device
    total_loss = []
    for j, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = train_helpers.prepare_batch(batch, device)
            imgs, target = batch["source"], batch["control"]
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
            loss = loss_spatial(pred, target)
            total_loss.append(loss.item())
            log_max = config.get("log_max")
            if log_max == -1 or j < log_max:
                target = train_helpers.standardize_feats(imgs, target)
                pred = train_helpers.standardize_feats(imgs, pred)
                grid = train_helpers.log_grid(imgs, target, pred)
                results_folder = train_helpers.make_results_folder(config, split, "preds", run_name=run_name)
                grid.save(f"{results_folder}/step-{step}_b-{j}.png")
            else:
                if split == "train":
                    break

def train(
    diffusion_extractor, 
    aggregation_network, 
    optimizer,
    train_dataloader, 
    val_dataloader,
    num_epochs,
    validation_epochs,
    device,
):

    aggregation_network = aggregation_network.to(device)

    for epoch in range(num_epochs):

        #################### VALIDATION ####################

        if epoch % validation_epochs == 0:
            with torch.no_grad():
                train_helpers.save_model(aggregation_network, optimizer, epoch)
                validate(diffusion_extractor, aggregation_network, train_dataloader, "train")
                validate(diffusion_extractor, aggregation_network, val_dataloader, "val")

        ##################### TRAINING #####################

        for batch in tqdm(train_dataloader):

            target, _, imgs = batch
            imgs, target = imgs.to(device), target.to(device)
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

    nsd = NaturalScenesDataset(
        root=config['data_root'],
        subject=config['subject'],
        partition="train",
        hemisphere=config['hemisphere'],
        roi=config['roi'],
    )
    dataset = NSDMeasuresDataset(
        nsd=nsd,
        measures=config['measures'],
        patches_shape=config['patches_shape'],
        img_shape=config['img_shape'],
        predict_average=True,
    )
    del nsd

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    train_dataloader = data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        shuffle=False,
    )

    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    
    config = {

        # General
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 0,
        "batch_size": 8,

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
        "data_root": "./data/NSD",
        "subject": 1,
        

        'results_folder': "./readout_results",
        

        "data_dir": "./data",
        "subject": 1,
        "roi": "floc-faces",
        "hemisphere": "right",
        "feature_extractor_type": "clip_2_0",
        "predict_average": True,
        "metric": "cosine",
        "n_neighbors": 0,

        "lr": 1e-4,
        
        "num_workers": 18,
        "num_epochs": 15,
        "validation_epochs": 1,

    }

    main(config)



