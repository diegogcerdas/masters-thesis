from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root,
        size,
    ):
        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        img = Image.open(self.instance_images_path[index % self.num_instance_images])
        return self.image_transforms(img)