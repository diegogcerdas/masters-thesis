from alignprop import run
import torch

if __name__ == "__main__":
    
    run(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_instance_prompt="a photo of <dgc>",
        args_num_timesteps=100,
        args_guidance_scale=7.5,
        args_lora_rank=4,
        args_lora_alpha=4,
        args_train_text_encoder=True,
        args_learning_rate=1e-4,
        args_batch_size=64,
        args_num_train_iterations=1000,
        args_validation_iters=100,
        args_validation_prompt="a photo of <dgc>",
        args_validation_images=25,
        args_brain_encoder_ckpt="./checkpoints/01_floc-faces_r_clip_1_5_0_cosine_avg_0",
        args_save_folder="./outputs/alignprop",
        args_seed=0,
        args_device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    )