from lora_slider import run_train
from lora_slider_test import run_test
import torch

if __name__ == "__main__":

    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    alpha=1
    rank=4
    training_method="full"
    train_steps=2500
    max_denoising_steps=50
    folder_main="./lora_slider"
    learning_rate=0.0002
    seed=42
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    save_path="./lora_slider/test"
    lora_weights_path="./lora_slider/test/last.pt"
    prompt = ""
    scales = [0, 1, 2, 3, 4, 5]
    start_noise = 800

    run_train(
        args_pretrained_model_name_or_path=pretrained_model_name_or_path,
        args_alpha=alpha,
        args_rank=rank,
        args_training_method=training_method,
        args_train_steps=train_steps,
        args_max_denoising_steps=max_denoising_steps,
        args_folder_main=folder_main,
        args_learning_rate=learning_rate,
        args_seed=seed,
        args_device=device,
        args_save_path=save_path,
    )
    run_test(
        args_pretrained_model_name_or_path=pretrained_model_name_or_path,
        args_alpha=alpha,
        args_rank=rank,
        args_training_method=training_method,
        args_lora_weights_path=lora_weights_path,
        args_prompt=prompt,
        args_scales=scales,
        args_start_noise=start_noise,
        args_seed=seed,
        args_device=device,
        args_save_folder=save_path,
    )