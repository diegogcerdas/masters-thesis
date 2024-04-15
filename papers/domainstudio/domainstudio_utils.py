import random
from pathlib import Path
import torch
import torch.utils.checkpoint
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_dir,
        instance_prompt,
        class_data_dir,
        class_prompt,
        num_class_images,
        tokenizer,
        size,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_images_path = [
            (x, instance_prompt)
            for x in Path(instance_data_dir).iterdir()
            if x.is_file() and not str(x).endswith(".txt")
        ]

        self.class_images_path = [
             (x, class_prompt) 
             for x in Path(class_data_dir).iterdir() 
             if x.is_file()
        ]
        self.class_images_path = self.class_images_path[:num_class_images]

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        instance_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_path).convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        class_path, class_prompt = self.class_images_path[index % self.num_class_images]
        class_image = Image.open(class_path).convert("RGB")

        example["class_images"] = self.image_transforms(class_image)
        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example
    

def collate_fn(examples, tokenizer):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    

def get_wave(x, pool=True):
        """wavelet decomposition using conv2d"""
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0) #  print(filter_LL.size())
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

        wave_channels = 3

        if pool:
            net = torch.nn.Conv2d
        else:
            net = torch.nn.ConvTranspose2d
        LL = net(wave_channels, wave_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=wave_channels)
        LH = net(wave_channels, wave_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=wave_channels)
        HL = net(wave_channels, wave_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=wave_channels)
        HH = net(wave_channels, wave_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=wave_channels)

        LL.weight.requires_grad = False
        LH.weight.requires_grad = False
        HL.weight.requires_grad = False
        HH.weight.requires_grad = False

        LL.weight.data = filter_LL.float().unsqueeze(0).expand(wave_channels, -1, -1, -1)
        LH.weight.data = filter_LH.float().unsqueeze(0).expand(wave_channels, -1, -1, -1)
        HL.weight.data = filter_HL.float().unsqueeze(0).expand(wave_channels, -1, -1, -1)
        HH.weight.data = filter_HH.float().unsqueeze(0).expand(wave_channels, -1, -1, -1)

        return LH.to(x.device), HL.to(x.device), HH.to(x.device)