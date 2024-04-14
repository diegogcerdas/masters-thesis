from domainstudio import run
import torch

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        resolution=512,
        num_timesteps=50,
        instance_prompt="a [V] animal",
        train_text_encoder=False,
        class_prompt="a animal",
        instance_data_dir="outputs/1_PPA_right/animal_not_person_cat_dog_bird_vehicle_furniture/min",
        class_data_dir="outputs/class/animal",
        num_class_images=200,
        validation_epochs=5,
        num_validation_images=10,
        validation_prompt="a [V] black bear",
        prior_loss_weight=1.0,
        image_loss_weight=1e+2,
        hf_loss_weight=1e+2,
        hfmse_loss_weight=0.1,
        learning_rate=5e-6,
        num_train_epochs=50,
        train_batch_size=4,
        outputs_dir="outputs/1_PPA_right/animal_not_person_cat_dog_bird_vehicle_furniture/min/domainstudio",
        seed=0,
        device=device,
    )