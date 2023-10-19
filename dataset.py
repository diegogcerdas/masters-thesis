from torch.utils.data import Dataset


class NaturalScenesDataset(Dataset):
    def __init__(self, partition: str):
        super().__init__()
        assert partition in ["train", "test"]

    def __len__(self):
        return NotImplementedError()

    def __getitem__(self, idx):
        return NotImplementedError()