from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class LoRADataset(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        resolution: int,
        class_data_root: str = None,
    ):
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.return_class_image = False
        if class_data_root is not None:
            self.return_class_image = True
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        instance_img = Image.open(self.instance_images_path[index % self.num_instance_images]).convert('RGB')
        instance_img = self.image_transforms(instance_img)
        if self.return_class_image:
            random_idx = torch.randint(0, self.num_class_images, (1,)).item()
            class_img = Image.open(self.class_images_path[random_idx]).convert('RGB')
            class_img = self.image_transforms(class_img)
            return instance_img, class_img
        return instance_img
