import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigTrain:
    subject: int
    roi: str
    hemisphere: str
    data_dir: str
    ckpt_dir: str
    logs_dir: str
    exp_name: str
    resume_ckpt: str
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int
    max_epochs: int
    device: str


def config_from_args(args, mode="train"):
    if mode == "train":
        class_name = ConfigTrain
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )
