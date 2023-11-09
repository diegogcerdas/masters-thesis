import random

import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RandomSpatialOffset:
    def __init__(self, offset, padding_mode="replicate"):
        self.offset = offset
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]

        dx = random.randint(-self.offset, self.offset)
        dy = random.randint(-self.offset, self.offset)

        # Perform edge value padding
        left, top, right, bottom = max(0, dx), max(0, dy), max(0, -dx), max(0, -dy)
        img = F.pad(img, (left, top, right, bottom), mode=self.padding_mode)

        # Crop back to original size
        left, top = max(0, -dx), max(0, -dy)
        img = TF.crop(img, top, left, h, w)

        return img
