import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigEncoder:
    subject: int
    roi: str
    hemisphere: str
    feature_extractor_type: str
    n_neighbors: int
    distance_metric: str
    predict_average: bool
    data_dir: str
    ckpt_dir: str
    logs_dir: str
    exp_name: str
    seed: int
    learning_rate: float
    batch_size: int
    num_workers: int
    max_epochs: int
    device: str
    wandb_project: str
    wandb_entity: str
    wandb_mode: str


@dataclass
class ConfigSynthesis:
    pretrained_model_name_or_path: str
    prompt: str
    num_images: int
    num_timesteps: int
    outputs_dir: str
    lora_dir: str
    seed: int
    device: str


@dataclass
class ConfigLora:
    pretrained_model_name_or_path: str
    data_dir: str
    instance_prompt: str
    num_timesteps: int
    lora_rank: int
    no_train_unet: bool
    no_train_text_encoder: bool
    validation_prompt: str
    validation_epochs: int
    num_val_images: int
    save_folder: str
    resolution: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    seed: int
    device: str


def config_from_args(args: dict, mode: str = "train"):
    if mode == "synthesis":
        class_name = ConfigSynthesis
    elif mode == "lora":
        class_name = ConfigLora
    elif mode == "encoder":
        class_name = ConfigEncoder
    else:
        raise ValueError("Mode not recognized.")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
