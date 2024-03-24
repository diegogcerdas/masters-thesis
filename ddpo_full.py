from ddpo import run
import torch

if __name__ == "__main__":
    
    run(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_lora_rank=4,
        args_train_text_encoder=True,
        args_brain_encoder_ckpt="./checkpoints/01_floc-faces_r_clip_1_5_0_cosine_avg_0/last.ckpt",
        args_prompt_filename="prompts.txt",
        args_guidance_scale=7.5,
        args_eta=1.0,
        args_num_timesteps=50,
        args_num_epochs=50,
        args_num_inner_epochs=10,
        args_validation_epochs=10,
        args_save_folder="./outputs/ddpo",
        args_adv_clip_max=5,
        args_clip_range=1e-4,
        args_batch_size=1,
        args_learning_rate=1e-4,
        args_seed=0,
        args_device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    )