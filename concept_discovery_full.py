import torch

from concept_discovery import run

if __name__ == "__main__":
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    run(
        args_pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
        args_seed=0,
        args_train_data_dir="./lora_slider/max",
        args_resolution=768,
        args_repeats=1,
        args_add_weight_per_score=True,
        args_init_weight=0.2,
        args_learning_rate=1e-3,
        args_train_batch_size=2,
        args_device=device,
        args_num_train_epochs=50,
        args_validation_epochs=5,
        args_output_dir="./lora_slider/concept_discovery",
        args_num_validation_images=10,
    )
