import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigSynthesis:
    pretrained_model_name_or_path: str
    prompt: str
    num_images: int
    inference_steps: int
    outputs_dir: str
    lora_dir: str
    seed: int
    device: str


@dataclass
class ConfigLora:
    data_dir: str
    subject: int
    hemisphere: str
    outputs_dir: str
    roi: str
    center_voxel: int
    n_neighbor_voxels: int
    voxels_filename: str
    feature_extractor_type: str
    distance_metric: str
    n_neighbors: int
    pos_std_threshold: float
    num_stimuli: int
    pretrained_model_name_or_path: str
    instance_prompt: str
    validation_prompt: str
    max_train_epochs: int
    num_validation_images: int
    validation_epochs: int
    train_text_encoder: bool
    inference_steps: int
    resolution: int
    rank: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    num_workers: int
    seed: int
    device: str


def config_from_args(args: dict, mode: str = "train"):
    if mode == "synthesis":
        class_name = ConfigSynthesis
    elif mode == "lora":
        class_name = ConfigLora
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
