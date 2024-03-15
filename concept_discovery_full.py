from concept_discovery import run
import torch

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    run(
        args_pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4',
        args_seed=0,
        args_train_data_dir='./concept_discovery/data',
        args_resolution=512,
        args_repeats=1,
        args_add_weight_per_score=True,
        args_init_weight=0.2,
        args_learning_rate=5e-3,
        args_train_batch_size=2,
        args_device=device,
        args_num_train_epochs=50,
        args_validation_epochs=10,
        args_output_dir='./concept_discovery/output',
        args_num_validation_images=10,
    )