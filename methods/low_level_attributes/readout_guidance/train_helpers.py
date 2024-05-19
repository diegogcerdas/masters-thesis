import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
from methods.low_level_attributes.readout_guidance.dhf.diffusion_extractor import DiffusionExtractor
from methods.low_level_attributes.readout_guidance.dhf.aggregation_network import AggregationNetwork
    

def load_models(config):
    diffusion_extractor = DiffusionExtractor(config, config['device'])
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=diffusion_extractor.dims,
        device=config['device'],
        save_timestep=config["save_timestep"],
        **config.get("aggregation_kwargs", {})
    )
    return diffusion_extractor, aggregation_network

def load_optimizer(config, aggregation_network):
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]},
    ]
    if config["aggregation_kwargs"].get("use_output_head", False):
        parameter_groups.append({"params": aggregation_network.output_head.parameters(), "lr": config["lr"]})
    optimizer = torch.optim.AdamW(parameter_groups)
    return optimizer

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=False):
    with torch.no_grad():
        feats, _ = diffusion_extractor.forward(imgs, eval_mode=eval_mode)
        b, _, _, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)), diffusion_extractor.emb)
    return diffusion_hyperfeats

def prepare_batch(batch, config):
    target, _, imgs = batch
    imgs, target = imgs.to(config["device"]), target.to(config["device"])
    imgs, target = resize_tensors(imgs, target, config["load_resolution"])
    imgs = renormalize(imgs, (0,1), (-1,1))
    return imgs, target

def resize_tensors(imgs, target, resolution):
    imgs = F.interpolate(imgs, resolution)
    target = F.interpolate(target, resolution)
    return imgs, target

def renormalize(x, range_a, range_b):
    # Note that if any value exceeds 255 in uint8 you get overflow
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def log_grid(imgs, target, pred):
    grid = []
    imgs = imgs.detach().cpu()
    imgs = renormalize(imgs, (-1, 1), (0, 1))
    grid.append(imgs)
    target = target.detach().cpu()
    target = renormalize(target, (target.min(), target.max()), (0, 1))
    grid.append(target)
    pred = pred.detach().cpu()
    pred = renormalize(pred, (pred.min(), pred.max()), (0, 1))
    grid.append(pred)
    grid = torch.cat(grid, dim=0)
    # Clamp to prevent overflow / underflow
    grid = torch.clamp(grid, 0, 1)
    grid = make_grid(grid, imgs.shape[0])
    return grid

def make_grid(images, nrow):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    grid = Image.fromarray(grid)
    return grid