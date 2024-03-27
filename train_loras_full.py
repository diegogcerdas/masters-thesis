from train_loras import run
import torch

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    run(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_data_dir='./outputs/PPA/cat',
        args_instance_prompt='a photo of a cat with background',
        args_num_timesteps=100,
        args_lora_rank=8,
        args_train_text_encoder=True,
        args_validation_prompt='a photo of a cat with background',
        args_validation_epochs=10,
        args_num_val_images=10,
        args_save_folder='./outputs/PPA/cat/loras',
        args_resolution=512,
        args_num_epochs=50,
        args_learning_rate=1e-4,
        args_seed=0,
        args_device=device,
    )