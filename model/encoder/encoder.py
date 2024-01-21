import torch.nn as nn


# TODO: Add support for other encoders
class EncoderType:
    LINEAR = "linear"


def create_encoder(type: EncoderType, input_size: int, output_size: int) -> nn.Module:
    if type == EncoderType.LINEAR:
        return nn.Linear(input_size, output_size)
