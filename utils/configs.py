import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigTrain:
    subject: int
    roi: str
    hemisphere: str
    feature_extractor_type: str
    encoder_type: str
    data_dir: str
    ckpt_dir: str
    logs_dir: str
    exp_name: str
    seed: int
    lr_start: float
    lr_end: float
    batch_size: int
    num_workers: int
    max_epochs: int
    device: str
    wandb_project: str
    wandb_entity: str
    wandb_mode: str


def config_from_args(args: dict, mode: str = "train"):
    if mode == "train":
        class_name = ConfigTrain
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
