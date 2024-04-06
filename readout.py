import torch
from tqdm import tqdm
import wandb
from readout_utils import load_models, load_optimizer, get_hyperfeats, log_aggregation_network, save_model
from datasets.nsd import NaturalScenesDataset
from datasets.nsd_features import NSDFeaturesDataset
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from torcheval.metrics.functional import r2_score
from torchvision import transforms


def validate(diffusion_extractor, aggregation_network, dataloader, epoch):
    device = aggregation_network.device
    total_loss = []
    total_metric = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            imgs, target, _ = batch
            imgs = transforms.Resize((config["resolution"], config["resolution"]))(imgs)
            imgs, target = imgs.to(device), target.to(device)
            pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
            loss = F.mse_loss(pred, target)
            metric = r2_score(pred, target)
            total_loss.append(loss.item())
            total_metric.append(metric.item())
    wandb.log({f"val/loss": np.mean(total_loss)}, step=epoch)
    wandb.log({f"val/metric": np.mean(total_metric)}, step=epoch)

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

    global_step = 0
    for epoch in range(num_epochs):

        #################### VALIDATION ####################

        if epoch % validation_epochs == 0:
            with torch.no_grad():
                fig = log_aggregation_network(aggregation_network, config)
                wandb.log({f"mixing_weights": fig}, step=epoch)
                save_model(config, aggregation_network, optimizer, global_step)
                validate(diffusion_extractor, aggregation_network, val_dataloader, epoch)

        #################### TRAINING ####################

        for batch in tqdm(train_dataloader):

            imgs, target, _ = batch
            imgs = transforms.Resize((config["resolution"], config["resolution"]))(imgs)
            imgs, target = imgs.to(device), target.to(device)
            pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs).squeeze()
            loss = F.mse_loss(pred, target)
            metric = r2_score(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/pred_mean": pred.mean().item()}, step=global_step)
            wandb.log({"train/pred_std": pred.std().item()}, step=global_step)
            wandb.log({"train/target_mean": target.mean().item()}, step=global_step)
            wandb.log({"train/target_std": target.std().item()}, step=global_step)
            
            wandb.log({"train/loss": loss.item()}, step=global_step)
            wandb.log({"train/metric": metric.item()}, step=global_step)
            wandb.log({"train/diffusion_timestep": diffusion_extractor.save_timestep[0]}, step=global_step)
            global_step += 1
            
        

if __name__ == "__main__":

    config = {
        "projection_dim": 384,
        "save_timestep": [0],
        'num_timesteps': 100,
        "aggregation_kwargs": {
            'use_output_head': True,
            'bottleneck_sequential': False,
        },
        'results_folder': "./readout_results",
        "model_id": "stabilityai/stable-diffusion-2",
        "resolution": 768,
        "prompt": "",
        "negative_prompt": "",
        "diffusion_mode": "generation",

        "data_dir": "./data",
        "subject": 1,
        "roi": "floc-faces",
        "hemisphere": "right",
        "feature_extractor_type": "clip_2_0",
        "predict_average": True,
        "metric": "cosine",
        "n_neighbors": 0,

        "lr": 1e-3,
        "batch_size": 8,
        "num_workers": 18,
        "num_epochs": 15,
        "validation_epochs": 1,
        "seed": 0,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        "wand_project": "readout",
        "wandb_entity": "diego-gcerdas",
        "wandb_mode": "online",
        "wandb_name": "test",
    }
    
    wandb.init(
        name=config['wandb_name'],
        project=config['wand_project'],
        entity=config['wandb_entity'],
        mode=config['wandb_mode'],
    )

    # Initialize dataset
    nsd = NaturalScenesDataset(
        root=config['data_dir'],
        subject=config['subject'],
        partition="train",
        roi=config['roi'],
        hemisphere=config['hemisphere'],
    )
    dataset = NSDFeaturesDataset(
        nsd=nsd,
        feature_extractor_type=config['feature_extractor_type'],
        predict_average=config['predict_average'],
        metric=config['metric'],
        n_neighbors=config['n_neighbors'],
        seed=config['seed'],
        device=config['device'],
    )
    del nsd

    config['aggregation_kwargs']['output_head_channels'] = dataset.target_size

    diffusion_extractor, aggregation_network = load_models(config)
    config["output_resolution"] = diffusion_extractor.output_resolution
    config["load_resolution"] = diffusion_extractor.load_resolution

    optimizer = load_optimizer(config, aggregation_network)

    # Initialize dataloaders (split into train and validation sets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    train_dataloader = data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=False,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=False,
        shuffle=False,
    )

    train(diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader, config['num_epochs'], config['validation_epochs'], config['device'])