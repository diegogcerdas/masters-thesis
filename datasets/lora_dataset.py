from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LoRADataset(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        resolution: int,
    ):
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
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
        img = Image.open(self.instance_images_path[index % self.num_instance_images]).convert('RGB')
        return self.image_transforms(img)
