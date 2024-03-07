import argparse
import torch
from utils.configs import config_from_args
from lora_dreambooth import run
from argparse import BooleanOptionalAction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Synthesis Parameters
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--instance_data_dir", type=str, default="./outputs/max/")
    parser.add_argument("--instance_prompt", type=str, default="sks")
    parser.add_argument("--validation_prompt", type=str, default="sks")
    parser.add_argument("--num_validation_images", type=int, default=10)
    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./outputs/lora/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_text_encoder", action=BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    args = parser.parse_args()
    cfg = config_from_args(args, mode="lora")

    run(
        args_pretrained_model_name_or_path=cfg.pretrained_path,
        args_instance_data_dir=cfg.instance_data_dir,
        args_instance_prompt=cfg.instance_prompt,
        args_validation_prompt=cfg.validation_prompt,
        args_num_validation_images=cfg.num_validation_images,
        args_validation_epochs=cfg.validation_epochs,
        args_output_dir=cfg.output_dir,
        args_seed=cfg.seed,
        args_train_text_encoder=cfg.train_text_encoder,
        args_train_batch_size=cfg.batch_size,
        args_max_train_epochs=cfg.max_train_epochs,
        args_learning_rate=cfg.learning_rate,
        args_lr_scheduler=cfg.lr_scheduler,
        args_dataloader_num_workers=cfg.num_workers,
        args_device=cfg.device,
    )

