import itertools
import random
from typing import List

import torch
from lightning import Fabric
from torch.utils.data import Sampler, Subset
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn

from abc import ABCMeta
from typing import Iterable, List, Sized


class SizedIterable(Sized, Iterable, metaclass=ABCMeta):
    pass


################################################################################
# create loaders
def create_field_loaders(
    config,
    split = "train",
    fabric = Fabric(),
    n_samples = None,
    sample_full_grid = False,
    setup_fabric = True,
    sample_points=True,
    shuffle = None,
    drop_last = True,
):
    """
    Creates data loaders for training, validation, or testing datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        split (str, optional): Dataset split to load. Options are "train", "val", or "test".
            Defaults to "train".
        fabric (Fabric, optional): Fabric object for distributed training.
            Defaults to a new Fabric instance.
        n_samples (int, optional): Number of samples to use for validation or testing.
            If None, defaults to 5000. Defaults to None.
        sample_full_grid (bool, optional): Whether to sample the full grid. Defaults to False.

    Returns:
        DataLoader: Configured DataLoader for the specified dataset split.
    """
    rebalance = (split == "train" and "rebalance" in config["dset"] and config["dset"]["rebalance"])

    rebalance = rebalance and ("use_single_dataset" not in config["dset"] or config["dset"]["use_single_dataset"] is None)
    dset = DatasetOmni(
        config,
        split=split,
        sample_points=sample_points,
        sample_full_grid=sample_full_grid,
        rebalance=rebalance,
    )

    if config["debug"] or split in ["val", "test"]:
        indexes = list(range(len(dset)))
        random.Random(0).shuffle(indexes)
        if n_samples is not None:
            indexes = indexes[:n_samples]
        else:
            indexes = indexes[:5000]
        if len(dset) > len(indexes):
            dset = Subset(dset, indexes)  # Smaller training set for debugging
    assert len(dset) > 0, f"{len(dset)=}"

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=(shuffle and not rebalance) if shuffle is not None else not rebalance if split == "train" else False,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=RandomizedMinorityUpsampler(dset.cluster_dict) if rebalance else None
    )
    fabric.print(f">> {split} set size: {len(dset)}")

    if setup_fabric:
        return fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
    return loader


def round_robin_shortest(iterables):
    min_len = min(len(iterable) for iterable in iterables)
    iterator_cycle = itertools.cycle(
        [
            itertools.cycle(iterable) if len(iterable) < min_len else iter(iterable)
            for iterable in iterables
            if len(iterable)
        ]
    )
    for iterator in iterator_cycle:
        try:
            yield next(iterator)
        except StopIteration:
            return


class RandomizedMinorityUpsampler(Sampler[int]):
    """Randomized version of upsampling shorter length lists of indices by cycling through them
    until the longer ones are exhausted."""

    def __init__(self, index_list: List[SizedIterable[int]]):
        self.index_list = index_list

    def __iter__(self):
        index_list = [idxlist.copy() for idxlist in self.index_list]
        random.shuffle(index_list)
        for idxlist_copy in index_list:
            random.shuffle(idxlist_copy)
        yield from round_robin_shortest(index_list)

    def __len__(self):
        min_len = min(len(idxlist) for idxlist in self.index_list)  # changed to min instead of max
        return min_len * len(self.index_list)
