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
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
