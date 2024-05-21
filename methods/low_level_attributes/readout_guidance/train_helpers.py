import torch
import torch.nn.functional as F
from PIL import Image
import io, os
import matplotlib.pyplot as plt
from methods.low_level_attributes.readout_guidance.dhf.diffusion_extractor import DiffusionExtractor
from methods.low_level_attributes.readout_guidance.dhf.aggregation_network import AggregationNetwork
    

def load_models(config):
    diffusion_extractor = DiffusionExtractor(config, config['device'])
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=diffusion_extractor.dims,
        device=config['device'],
        save_timestep=config["save_timestep"],
        use_output_head=config["use_output_head"],
        output_head_channels=config["output_head_channels"],
        bottleneck_sequential=config["bottleneck_sequential"],
    )
    return diffusion_extractor, aggregation_network

def load_optimizer(config, aggregation_network):
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]},
    ]
    if config["use_output_head"]:
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
    imgs, target = batch
    imgs, target = imgs.to(config["device"]), target.to(config["device"])
    imgs, target = resize_tensors(imgs, target, config)
    imgs = renormalize(imgs, (0,1), (-1,1))
    return imgs, target

def resize_tensors(imgs, target, config):
    imgs = F.interpolate(imgs, config["load_resolution"])
    target = F.interpolate(target, config["output_resolution"])
    return imgs, target

def renormalize(x, range_a, range_b):
    # Note that if any value exceeds 255 in uint8 you get overflow
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def log_grid(imgs, target, pred):
    num_images = min(16, imgs.shape[0])
    min_val = min(target.min(), pred.min())
    max_val = max(target.max(), pred.max())
    fig, axes = plt.subplots(3, num_images, figsize=(num_images*3, 9))
    for i in range(num_images):
        img = F.interpolate(imgs[i].detach().cpu(), 128).permute(1, 2, 0)
        img = renormalize(img, (-1, 1), (0, 1))
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[1, i].imshow(target[i].detach().cpu().permute(1, 2, 0), vmin=min_val, vmax=max_val, cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(pred[i].detach().cpu().permute(1, 2, 0), vmin=min_val, vmax=max_val, cmap='gray')
        axes[2, i].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def save_model(config, aggregation_network, epoch):
    ckpt_folder = os.path.join(config['results_folder'], config['exp_name'], 'checkpoints')
    os.makedirs(ckpt_folder, exist_ok=True)
    torch.save(aggregation_network.state_dict(), os.path.join(ckpt_folder, f"epoch-{str(epoch).zfill(3)}.pt"))
    torch.save(aggregation_network.state_dict(), os.path.join(ckpt_folder, f"last.pt"))