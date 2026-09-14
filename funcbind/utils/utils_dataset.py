import inspect
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


# `in_order` is a DataLoader argument only from torch 2.6; on 2.5 and older, passing it
# raises TypeError before a single batch is read. Probe the signature instead of parsing
# a version string so a backport or a fork is judged by what it actually accepts.
_DATALOADER_ACCEPTS_IN_ORDER = "in_order" in inspect.signature(
    torch.utils.data.DataLoader.__init__
).parameters
_warned_no_in_order = False


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

    num_workers = int(config["dset"]["num_workers"])
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(config["dset"].get("persistent_workers", True)),
            prefetch_factor=int(config["dset"].get("prefetch_factor", 4)),
        )
        # Training samples can vary substantially in CPU crop/resampling cost.
        # Consume the first ready worker result instead of letting one slow crop
        # block already-prepared batches behind it. Keep evaluation deterministic.
        want_in_order = (
            bool(config["dset"].get("in_order", True)) if split == "train" else True
        )
        if _DATALOADER_ACCEPTS_IN_ORDER:
            loader_kwargs["in_order"] = want_in_order
        elif not want_in_order:
            # torch < 2.6 has no such knob and is always in-order. Say so once: the
            # config asked for out-of-order consumption and will not get it, which
            # costs throughput when one crop straggles.
            global _warned_no_in_order
            if not _warned_no_in_order:
                _warned_no_in_order = True
                fabric.print(
                    f">> NOTE: dset.in_order=False ignored — torch {torch.__version__} "
                    "has no DataLoader(in_order=...); needs torch >= 2.6. "
                    "Loading stays in-order."
                )

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=num_workers,
        shuffle=(shuffle and not rebalance) if shuffle is not None else not rebalance if split == "train" else False,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=RandomizedMinorityUpsampler(dset.cluster_dict) if rebalance else None,
        **loader_kwargs,
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
