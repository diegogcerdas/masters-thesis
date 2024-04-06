from dhf.diffusion_extractor import DiffusionExtractor
from dhf.aggregation_network import AggregationNetwork
import torch
import matplotlib.pyplot as plt
import os

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

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    else:
        save_timestep = [0]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    return fig

def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = f"{config['results_folder']}/{step}"
    os.makedirs(results_folder, exist_ok=True)
    torch.save(dict_to_save, f"{results_folder}/ckpt.pt")