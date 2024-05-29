import argparse
import os
from argparse import BooleanOptionalAction

import torch
import torch.nn.functional as F
import wandb
from torch.utils import data
from tqdm import tqdm

from datasets.nsd.nsd import NaturalScenesDataset
from datasets.nsd.nsd_measures import NSDMeasuresDataset
from methods.low_level_attributes.readout_guidance.train_helpers import (
    get_hyperfeats, load_models, load_optimizer, log_grid, prepare_batch,
    save_model)


def train(
    config,
    diffusion_extractor, 
    aggregation_network, 
    optimizer,
    train_dataloader, 
    val_dataloader,
):

    aggregation_network = aggregation_network.to(config["device"])

    global_step = 0
    for epoch in range(config["num_epochs"]):

        ##################### TRAINING #####################

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
  
            imgs, target = prepare_batch(batch, config)
            pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = F.mse_loss(pred, target)
            wandb.log({"train/loss": loss.item()}, step=global_step)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        #################### VALIDATION ####################

        if epoch % config["validation_epochs"] == 0:
            
            total_loss = 0
            with torch.no_grad():

                for j, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    
                    imgs, target = prepare_batch(batch, config)
                    pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
                    total_loss += F.mse_loss(pred, target).item()

                    if j <= config["log_max"]:
                        grid = log_grid(imgs, target, pred, config["dataset_args"]["control_range"])
                        grid.save(os.path.join(config["results_folder"], config["exp_name"], f"epoch-{epoch}_b-{j}.png"))
                
                save_model(config, aggregation_network, epoch)
            
            wandb.log({"val/loss": total_loss/len(val_dataloader)}, step=global_step)
        

def main(config):

    diffusion_extractor, aggregation_network = load_models(config)
    config["output_resolution"] = diffusion_extractor.output_resolution
    config["load_resolution"] = diffusion_extractor.load_resolution
    config["dataset_args"] = {"control_range": (-5,5)}
    config["dims"] = diffusion_extractor.dims

    optimizer = load_optimizer(config, aggregation_network)
    
    # Training set
    training_sets = []
    for subject in [1,2,3,4,5,6,7,8]:
        nsd = NaturalScenesDataset(
            root=config['dataset_root'],
            subject=subject,
            partition="train",
        )
        nsd_measures = NSDMeasuresDataset(
            nsd=nsd,
            measures=config['measures'],
            patches_shape=config['patches_shape'],
            img_shape=config['img_shape'],
        )
        training_sets.append(nsd_measures)
    train_set = data.ConcatDataset(training_sets)
    train_dataloader = data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    # Validation set
    validation_sets = []
    for subject in [1,2,3,4,5,6,7,8]:
        nsd = NaturalScenesDataset(
            root=config['dataset_root'],
            subject=subject,
            partition="test",
        )
        nsd_measures = NSDMeasuresDataset(
            nsd=nsd,
            measures=config['measures'],
            patches_shape=config['patches_shape'],
            img_shape=config['img_shape'],
        )
        validation_sets.append(nsd_measures)
    val_set = data.ConcatDataset(validation_sets)
    val_dataloader = data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    config["exp_name"] = config["measures"]
    os.makedirs(os.path.join(config["results_folder"], config["exp_name"]), exist_ok=True)

    wandb.init(
        name=config["exp_name"],
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        mode=config["wandb_mode"],
    )

    train(
        config,
        diffusion_extractor, 
        aggregation_network, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Aggregation Network Parameters
    parser.add_argument("--projection_dim", type=int, default=384)
    parser.add_argument("--use_output_head", action=BooleanOptionalAction, default=True)
    parser.add_argument("--output_head_channels", type=int, default=1)
    parser.add_argument("--bottleneck_sequential", action=BooleanOptionalAction, default=False)

    # Diffusion Extractor Parameters
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--save_timestep", type=int, nargs="+", default=[0])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--diffusion_mode", type=str, default="generation")

    # Dataset Parameters
    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--measures", type=str, default="warmth")
    parser.add_argument("--patches_shape", type=tuple, default=(64, 64))
    parser.add_argument("--img_shape", type=tuple, default=(448, 448))

    # General Parameters
    parser.add_argument("--results_folder", type=str, default="./data/readout_guidance/")
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_max", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # WandB Parameters
    parser.add_argument("--wandb-project", type=str, default="masters-thesis-readout")
    parser.add_argument("--wandb-entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb-mode", type=str, default="online")

    args = parser.parse_args()
    config = vars(args)
    main(config)