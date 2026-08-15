import functools
import os
import random
import sys

import torch
from torch.utils.data import Dataset


@functools.lru_cache(maxsize=4)
def _voxbind_density_ops(voxbind_root: str):
    """(rotate, translate) for density volumes, borrowed from VoxBind.

    Reused rather than reimplemented so the augmentation applied to a crop here is
    bit-for-bit the one VoxBind trains with; a subtly different interpolation would
    make the two repos' density conditioning quietly incomparable.
    """
    if voxbind_root not in sys.path:
        sys.path.insert(0, voxbind_root)
    from voxbind.dataset.crossdocked_xray import _rotate_density, _translate_density

    return _rotate_density, _translate_density


class DatasetReceptorLigand(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/data/",
        input_dataset: str = "crossdocked_pocket10",
        split: str = "train",
        small: bool = False,
        val_from_train_pool: bool = False,
        val_size: int = 100,
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
            val_sz = val_size
            pool_n = len(data) - val_sz  # rows VoxBind precomputed density crops for

            if val_from_train_pool:
                # VoxBind convention: hold val out from INSIDE the crop-aligned pool, so
                # both splits have an electron-density crop. The default (tail) val sits
                # at indices >= pool_n, which no crop file covers.
                n_train = pool_n - val_sz
                idx = range(n_train) if split == "train" else range(n_train, pool_n)
            else:
                idx = range(pool_n) if split == "train" else range(pool_n, len(data))

            self.source_index = list(idx)
            self.data = [data[i] for i in idx]
        else:
            file = os.path.join(self.data_dir, "data_test.pt")
            self.data = torch.load(file)
            self.source_index = list(range(len(self.data)))

        # filter dataset
        if self.small:
            self.data = self.data[:500]
            self.source_index = self.source_index[:500]
        # source_index[i] is the row's position in the shuffled data_train.pt, i.e. the
        # name of its density crop ({source_index[i]:06d}.npy). Keep them in lockstep.
        assert len(self.source_index) == len(self.data)
        print("Loaded dataset", file, "with size", len(self.data),
              f"(split={split}, val_from_train_pool={val_from_train_pool})")
