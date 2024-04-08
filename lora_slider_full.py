import torch
import numpy as np

from lora_slider import run_train

if __name__ == "__main__":
    
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
    resolution = 768
    alpha = 8
    rank = 8
    training_method = "full"
    num_timesteps = 50
    folder_main = "./lora_slider"
    save_path = "./lora_slider/results"
    num_epochs = 50
    learning_rate = 1e-4
    seed = 0
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    validation_scales = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    validation_prompt = ""
    validation_guidance_scale = 1
    validation_start_noise = 40
    validation_epochs = 10

    run_train(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        resolution=resolution,
        alpha=alpha,
        rank=rank,
        scales=np.array([-1, 1]),
        folders=np.array(["null", "max"]),
        training_method=training_method,
        num_timesteps=num_timesteps,
        folder_main=folder_main,
        save_path=save_path,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        validation_scales=validation_scales,
        validation_prompt=validation_prompt,
        validation_guidance_scale=validation_guidance_scale,
        validation_start_noise=validation_start_noise,
        validation_epochs=validation_epochs,
        seed=seed,
        device=device,
    )
