import os
import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def freeze_params(params):
    for param in params:
        param.requires_grad = False


class ComposableDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size,
        repeats,
        placeholder_tokens,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.repeats = repeats
        self.placeholder_tokens = placeholder_tokens

        self.image_paths = [
            os.path.join(data_root, file_name)
            for file_name in os.listdir(data_root)
            if file_name.endswith(".png")
        ]
        self.num_images = len(self.image_paths)

    def __len__(self):
        return self.num_images * self.repeats

    def __getitem__(self, i):
        idx = i % self.num_images
        example = dict()

        image = Image.open(self.image_paths[idx]).resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        text = self.placeholder_tokens  # use token itself as the caption (unsupervised)
        # encode all classes since we will use all of them to compute composed score
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["pixel_values"] = image
        example["gt_weight_id"] = idx
        example["image_path"] = self.image_paths[idx]
        example["image_index"] = idx
        return example
