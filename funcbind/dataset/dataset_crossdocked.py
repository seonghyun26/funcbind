import os
import random

import torch
from torch.utils.data import Dataset


class DatasetReceptorLigand(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/data/",
        input_dataset: str = "crossdocked_pocket10",
        split: str = "train",
        small: bool = False,
    ):
        assert split in ["train", "val", "test"]

        self.data_dir = os.path.join(data_dir, input_dataset)
        self.split = split
        self.small = small

        # Read data
        if split == "train" or split == "val":
            file = os.path.join(self.data_dir, "data_train.pt")
            data = torch.load(file, weights_only=False)
            random.Random(1234).shuffle(data)
            val_sz = 100
            self.data = data[: (len(data) - val_sz)] if split == "train" else data[(len(data) - val_sz) :]
        else:
            file = os.path.join(self.data_dir, "data_test.pt")
            self.data = torch.load(file)

        # filter dataset
        self.data = self.data[:500] if self.small else self.data
        print("Loaded dataset", file, "with size", len(self.data))
