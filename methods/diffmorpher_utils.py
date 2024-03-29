import bisect

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def distance(img_a, img_b):
    return lpips.LPIPS()(img_a, img_b).item()

class AlphaScheduler:

    def __init__(self, images_pt):
        self.__num_values = len(images_pt)
        self.__values = [0]
        for i in range(self.__num_values - 1):
            dis = distance(images_pt[i], images_pt[i + 1])
            self.__values.append(dis)
            self.__values[i + 1] += self.__values[i]
        for i in range(self.__num_values):
            self.__values[i] /= self.__values[-1]

    def get_x(self, y):
        assert y >= 0 and y <= 1
        id = bisect.bisect_left(self.__values, y)
        id -= 1
        if id < 0:
            id = 0
        yl = self.__values[id]
        yr = self.__values[id + 1]
        xl = id * (1 / (self.__num_values - 1))
        xr = (id + 1) * (1 / (self.__num_values - 1))
        x = (y - yl) / (yr - yl) * (xr - xl) + xl
        return x

    def get_list(self, len=None):
        if len is None:
            len = self.__num_values

        ys = torch.linspace(0, 1, len)
        res = [self.get_x(y) for y in ys]
        return res
    

def get_latents(pipeline, batch_size, generator, dtype, device):
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator
    )
    return latents

@torch.no_grad()
def get_text_embeddings(tokenizer, text_encoder, prompt, device):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=77, return_tensors="pt"
    ).to(device)
    text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    return text_embeddings