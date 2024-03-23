import os

import torch

from diffmorpher import run

if __name__ == "__main__":
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    images, alpha_list = run(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_img_path_0="./diffmorpher/data/0.png",
        args_img_path_1="./diffmorpher/data/1.png",
        args_load_lora_path_0="./outputs/max_0/outputs/final",
        args_load_lora_path_1="./outputs/max_0_null/outputs/final",
        args_prompt_0="",
        args_prompt_1="",
        args_num_inference_steps=250,
        args_num_frames=50,
        args_use_adain=True,
        args_use_reschedule=True,
        args_resolution=512,
        args_seed=42,
        args_device=device,
    )

    out_dir = "./outputs/max_0_results"
    os.makedirs(out_dir, exist_ok=True)
    for img, alpha in zip(images, alpha_list):
        img.save(f"{out_dir}/{alpha:.2f}.png")
