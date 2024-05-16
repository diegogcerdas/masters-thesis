import torch
from dhf.diffusion_extractor import DiffusionExtractor
from dhf.aggregation_network import AggregationNetwork
    

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
        b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)), diffusion_extractor.emb)
    return diffusion_hyperfeats