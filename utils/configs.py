import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigTrain:
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
class ConfigPipeline:
    data_dir: str
    subject: int
    hemisphere: str
    roi: str
    center_voxel: int
    n_neighbor_voxels: int
    voxels_filename: str
    feature_extractor_type: str
    distance_metric: str
    n_neighbors: int
    neg_std_threshold: float
    pos_std_threshold: float
    num_captioned: float
    prompt_clip_model: str
    prompt_clip_pretrain: str
    prompt_iterations: int
    prompt_lr: float
    prompt_weight_decay: float
    prompt_prompt_len: int
    prompt_prompt_bs: int
    prompt_loss_weight: float
    prompt_batch_size: int
    outputs_dir: str
    slerp_steps: int
    g: float
    inference_steps: int
    seed: int
    device: str


@dataclass
class ConfigSynthesis:
    prompt: str
    seed: int
    g: float
    inference_steps: int
    outputs_dir: str
    batch_size: int
    device: str


@dataclass
class ConfigSynthesis2:
    prompt1: str
    prompt2: str
    steps: int
    seed: int
    g: float
    inference_steps: int
    outputs_dir: str
    device: str


def config_from_args(args: dict, mode: str = "train"):
    if mode == "train":
        class_name = ConfigTrain
    elif mode == "synthesis":
        class_name = ConfigSynthesis
    elif mode == "synthesis2":
        class_name = ConfigSynthesis2
    elif mode == "pipeline":
        class_name = ConfigPipeline
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
