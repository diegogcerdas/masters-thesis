import torch.nn as nn


# TODO: Add support for other encoders
class EncoderType:
    LINEAR = "linear"
    MLP = "mlp"


def create_encoder(type: EncoderType, input_size: int, output_size: int) -> nn.Module:
    if type == EncoderType.LINEAR:
        return nn.Linear(input_size, output_size)
    elif type == EncoderType.MLP:
        return nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
