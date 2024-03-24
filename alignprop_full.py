from alignprop import run
import torch

if __name__ == "__main__":
    
    run(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_prompt_filename="prompt.txt",
        args_num_timesteps=50,
        args_guidance_scale=5,
        args_lora_rank=10,
        args_train_text_encoder=True,
        args_learning_rate=1e-4,
        args_batch_size=1,
        args_num_train_iterations=500,
        args_validation_iters=50,
        args_validation_prompt="",
        args_validation_images=10,
        args_brain_encoder_ckpt="./checkpoints/01_floc-faces_r_clip_1_5_0_cosine_avg_0/last.ckpt",
        args_save_folder="./outputs/alignprop",
        args_seed=0,
        args_device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    )