import argparse
import torch
from utils.configs import config_from_args
from utils.img_utils import save_images
from utils.configs import ConfigInterpolation
from methods.lora_interpolate import interpolate

def main(cfg: ConfigInterpolation):

    images, alpha_list = interpolate(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        lora_min_path=cfg.lora_min_path,
        lora_max_path=cfg.lora_max_path,
        prompt=cfg.prompt,
        num_timesteps=cfg.num_timesteps,
        num_frames=cfg.num_frames,
        seed=cfg.seed,
        device=cfg.device,
    )

    # save the images
    names = [f'{alpha:.2f}' for alpha in alpha_list]
    save_images(images, cfg.outputs_dir, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained-model-name-or-path", type=str, default="stabilityai/stable-diffusion-2",)
    parser.add_argument("--lora-min-path", type=str, default="./outputs/1_PPA_right/food_not_person_animal/lora/min/final")
    parser.add_argument("--lora-max-path", type=str, default="./outputs/1_PPA_right/food_not_person_animal/lora/max/final")
    parser.add_argument("--prompt", type=str, default="a photo of food with background")
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--outputs-dir", type=str, default="./outputs/1_PPA_right/food_not_person_animal/interpolate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="interpolation")
    main(cfg)