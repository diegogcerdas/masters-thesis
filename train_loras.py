import argparse
import torch, os
from methods.lora_training import train_lora
from utils.configs import config_from_args, ConfigLora
from argparse import BooleanOptionalAction

def main(cfg: ConfigLora):

    for folder in ['min', 'max']:

        data_dir = os.path.join(cfg.data_dir, folder)
        save_folder = os.path.join(cfg.save_folder, folder)
    
        train_lora(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            data_dir=data_dir,
            instance_prompt=cfg.instance_prompt,
            num_timesteps=cfg.num_timesteps,
            lora_rank=cfg.lora_rank,
            omit_unet=cfg.omit_unet,
            omit_text_encoder=cfg.omit_text_encoder,
            validation_prompt=cfg.validation_prompt,
            validation_epochs=cfg.validation_epochs,
            num_val_images=cfg.num_val_images,
            save_folder=save_folder,
            resolution=cfg.resolution,
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            device=cfg.device,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained-model-name-or-path", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--data-dir", type=str, default="./outputs/1_PPA_right/food_not_person_animal")
    parser.add_argument("--instance-prompt", type=str, default="a photo of food with background")
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--omit-unet", action=BooleanOptionalAction, default=False)
    parser.add_argument("--omit-text-encoder", action=BooleanOptionalAction, default=False)
    parser.add_argument("--validation-prompt", type=str, default="a photo of food with background")
    parser.add_argument("--validation-epochs", type=int, default=10)
    parser.add_argument("--num-val-images", type=int, default=15)
    parser.add_argument("--save-folder", type=str, default="./outputs/1_PPA_right/food_not_person_animal/lora")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="lora")
    main(cfg)