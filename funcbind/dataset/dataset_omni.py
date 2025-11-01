import hydra
import os
import itertools

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

from funcbind.dataset.dataset_crossdocked import DatasetReceptorLigand
from funcbind.dataset.dataset_mcpp import DatasetMCPP
from funcbind.dataset.dataset_ab import DatasetAntibodyAntigen
from funcbind.utils.constants import PADDING_INDEX
from funcbind.models.decoder import get_grid
import numpy as np
from funcbind.utils.utils_base import (
    atomChannelsToRadius,
    filter_atoms_by_distance_mask,
    recenter_structures,
    rotate_coords,
    translate_coords,
)


class DatasetOmni(Dataset):
    def __init__(
        self,
        config,
        split= "train",
        delta_translate = 1.0,
        sample_points = True,
        sample_full_grid = False,
        rebalance = True,
    ):
        # TODO: this use_single_dataset is kinda weird. We should just pass a list of datasets
        # that we want to consider in OMNI. If we want to use single dataset, just pass one of them
        # Change that once we have some time...
        assert split in ["train", "val", "test"]
        assert "use_single_dataset" not in config["dset"]  or config["dset"]["use_single_dataset"] in [None, "xdocked", "mcpp", "sabdab"]

        self.split = split
        self.ligand_radius = config["dset"]["ligand_radius"]
        self.receptor_radius = config["dset"]["ligand_radius"] if config["dset"]["same_radius"] else config["dset"]["receptor_radius"]
        self.aug = config["dset"]["data_aug"] if split == "train" else False
        self.delta = delta_translate
        self.grid_dim = config["dset"]["grid_dim"]
        self.resolution = config["dset"]["resolution"]
        self.max_dim = (self.grid_dim * self.resolution) // 2
        self.sample_points = sample_points
        self.sample_full_grid = sample_full_grid
        self.n_points = config["dset"]["n_points"]
        self.use_single_dataset = None if "use_single_dataset" not in config["dset"] else config["dset"]["use_single_dataset"]
        self.rebalance = rebalance

        # Field params
        self.targeted_sampling_ratio = config["dset"]["targeted_sampling_ratio"] if split == "train" else 0
        self.discrete_grid, self.full_grid_high_res = get_grid(self.grid_dim)
        self.cubes_around = config["dset"]["cubes_around"]
        self.increments = torch.tensor(
            list(itertools.product(list(range(-self.cubes_around, self.cubes_around+1)), repeat=3))
        )

        self.data = []
        data_xdocked = []
        data_mcpp = []
        data_sabdab = []

        # load xdocked
        if self.use_single_dataset is None or self.use_single_dataset == "xdocked":
            data_xdocked = DatasetReceptorLigand(
                data_dir=config["dset"]["data_dir"],
                input_dataset="crossdocked_pocket10",
                split=split,
            ).data
            self.data.extend(data_xdocked)

        # load mcpp
        if self.use_single_dataset is None or self.use_single_dataset == "mcpp":
            data_mcpp = DatasetMCPP(
                data_dir=config["dset"]["data_dir"],
                input_dataset="mcpp_dataset",
                split=split,
                elements=config["dset"]["elements"],
            ).data
            self.data.extend(data_mcpp)

        # load sabdab
        if self.use_single_dataset is None or self.use_single_dataset == "sabdab":
            data_sabdab = DatasetAntibodyAntigen(
                data_dir=config["dset"]["data_dir"],
                dataset_name=config["dset"]["input_dataset"] if "omni" not in config["dset"]["input_dataset"] else "sabdab_v0.5.2_diffab_chothia",
                split=split,
                cdrs_aug=config["dset"]["cdrs_aug"],
                cdrs=["H3"] if "cdrs" not in config["dset"] else config["dset"]["cdrs"],
                grid_dim=self.grid_dim,
                resolution=self.resolution,
            ).data
            self.data.extend(data_sabdab)

        self.range_xdocked = list(range(len(data_xdocked)))
        self.range_mcpp = list(range(len(data_xdocked), len(data_xdocked) + len(data_mcpp)))
        self.range_sabdab = list(range(len(data_xdocked)+len(data_mcpp), len(data_xdocked)+len(data_mcpp)+len(data_sabdab)))
        if self.rebalance:
            self.cluster_dict = [self.range_xdocked] + [self.range_mcpp] + [self.range_sabdab]


    def __len__(self) -> int:
        if self.rebalance:
            min_len = min(len(idxlist) for idxlist in self.cluster_dict)
            return min_len * len(self.cluster_dict)
        return len(self.data)


    def preprocess_xdocked_and_mcpp(self, sample, dset):
        if dset == "xdocked":
            receptor_, ligand_ = sample
            ligand_id, receptor_id = ligand_["id"], receptor_["id"]
            data_type = 0
        else:
            receptor_, ligand_ = sample[1], sample[0]
            ligand_id =  "/".join(sample[2].split("/")[-3:]).replace(".sdf", ".pdb")
            receptor_id = ligand_id.replace("-MCP.", "-protein.") if "MCP." in ligand_id else ligand_id.replace("-CP.", "-protein.")
            data_type = 1

        # ligand
        if not isinstance(ligand_["atoms_channel"], torch.Tensor):
            ligand_["atoms_channel"] = torch.from_numpy(ligand_["atoms_channel"])
            ligand_["coords"] = torch.from_numpy(ligand_["coords"])
        mask = ligand_["atoms_channel"] < 8
        if self.ligand_radius > 0:
            radius = self.ligand_radius * torch.ones_like(ligand_["atoms_channel"][mask]).float()
        else:
            radius = atomChannelsToRadius(ligand_["atoms_channel"][mask])
        ligand = {
            "coords": ligand_["coords"][mask],
            "atoms_channel": ligand_["atoms_channel"][mask].float(),
            "radius": radius,
            "id": ligand_id,
            "cdr_h3_seq": "",
            "data_type": data_type
        }
        center_coords = ligand["coords"].mean(axis=0)

        # receptor
        if not isinstance(receptor_["atoms_channel"], torch.Tensor):
            receptor_["atoms_channel"] = torch.from_numpy(receptor_["atoms_channel"])
            receptor_["coords"] = torch.from_numpy(receptor_["coords"])
        mask = receptor_["atoms_channel"] < 4  # receptor only has C, O, N, S
        if self.receptor_radius > 0:
            radius = self.receptor_radius * torch.ones_like(receptor_["atoms_channel"][mask]).float()
        else:
            radius = atomChannelsToRadius(receptor_["atoms_channel"][mask])
        receptor = {
            "coords": receptor_["coords"][mask],
            "atoms_channel": receptor_["atoms_channel"][mask].float(),
            "radius": radius,
            "center_coords": center_coords,
            "id": receptor_id,
            "data_type": data_type
        }

        return ligand, receptor

    def preprocess_sabdab(self, sample):
        ag, ab = sample[0], sample[1]
        cdr_mask = ab["cdr_mask"]
        if self.ligand_radius > 0:
            ab['radius'] = self.ligand_radius * torch.ones_like(ab["atoms_channel"] ).float()
        else:
            ab['radius'] = atomChannelsToRadius(ab["atoms_channel"])

        # Process Ag
        if ag is not None:
            if self.receptor_radius > 0:
                ag['radius'] = self.receptor_radius * torch.ones_like(ag["atoms_channel"]).float()
            else:
                ag['radius'] = atomChannelsToRadius(ag["atoms_channel"])
            # Include non CDR Ab regions into receptor
            ag = {
                "coords": torch.cat([ag["coords"], ab["coords"][~cdr_mask]], dim=0),
                "atoms_channel": torch.cat([ag["atoms_channel"], ab["atoms_channel"][~cdr_mask]], dim=0).float(),
                "radius": torch.cat([ag["radius"], ab["radius"][~cdr_mask]], dim=0),
                "id": ag["id"],
                "data_type": 2,
            }
        else:
            # Only non CDR Ab regions into receptor
            ag = {
                "coords": ab["coords"][~cdr_mask],
                "atoms_channel": ab["atoms_channel"][~cdr_mask].float(),
                "radius": ab["radius"][~cdr_mask],
                "id": ab["id"],
                "data_type": 2,
            }

        # Process ab
        ab_ = {
            "coords": ab["coords"][cdr_mask],
            "atoms_channel": ab["atoms_channel"][cdr_mask].float(),
            "radius": ab["radius"][cdr_mask],
            "id": ab["id"],
            "cdr_h3_seq": ab["cdr_h3_seq"],
            "data_type": 2,
        }

        ag["center_coords"] = ab_["coords"].mean(axis=0)

        return ab_, ag

    def get_ligand_receptor(self, index):
        sample = self.data[index]
        if index in self.range_xdocked:
            ligand, receptor = self.preprocess_xdocked_and_mcpp(sample, dset="xdocked")
        elif index in self.range_mcpp:
            ligand, receptor = self.preprocess_xdocked_and_mcpp(sample, dset="mcpp")
        elif index in self.range_sabdab:
            ligand, receptor = self.preprocess_sabdab(sample)
        return ligand, receptor

    def _get_xs(self, sample) -> torch.Tensor:
        """
        Get the point coordinates for the grid.
        """
        coords = sample["coords"]
        if self.sample_full_grid:
            xs = self.full_grid_high_res
        else:
            if self.targeted_sampling_ratio >= 1:  # non-uniform sampling of space (upsample coords close to center of each atom)
                rand_points = (coords / self.resolution).long()
                num_random_elements = max(1, (self.n_points // rand_points.shape[0]) // self.targeted_sampling_ratio)
                random_indices = torch.randperm(self.increments.size(0))[:num_random_elements]
                rand_points = (rand_points.unsqueeze(1) + self.increments[random_indices].unsqueeze(0)).reshape(-1, 3)
                rand_points = torch.clamp(rand_points, -self.grid_dim // 2, self.grid_dim // 2) / (self.grid_dim // 2)
                grid_points = torch.Tensor(np.random.choice(self.discrete_grid, (self.n_points - rand_points.shape[0], 3)))
                xs = torch.cat([rand_points, grid_points], dim=0)
            else:  # uniform sampling of space
                xs = torch.Tensor(np.random.choice(self.discrete_grid, (self.n_points, 3)))
        return xs

    def __getitem__(self, index) -> dict:
        ligand, receptor = self.get_ligand_receptor(index)

        # center reference of frame to center of mass of ligand
        ligand, receptor = recenter_structures(ligand, receptor, receptor["center_coords"])

        # add aug (rotation then translation)
        if self.aug:
            ligand, receptor = rotate_coords(ligand, receptor)
            ligand, receptor = translate_coords(ligand, receptor, self.delta)

        # box molecules
        ligand = filter_atoms_by_distance_mask(ligand, max_dim=self.max_dim)
        if receptor is not None:
            receptor = filter_atoms_by_distance_mask(receptor, max_dim=self.max_dim)

        # add xs
        if self.sample_points:
            ligand.update({"xs": self._get_xs(ligand)})

        return {"receptor": receptor, "ligand": ligand}


def collate_fn(batch):
    # Pad in the batch
    has_receptor = "receptor" in batch[0]
    if has_receptor:
        id_ = default_collate([item["receptor"]["id"] for item in batch])
        coords = [item["receptor"]["coords"] for item in batch]
        coords = torch.nn.utils.rnn.pad_sequence(
            coords, batch_first=True, padding_value=PADDING_INDEX
        )
        atoms_channel = [item["receptor"]["atoms_channel"] for item in batch]
        atoms_channel = torch.nn.utils.rnn.pad_sequence(
            atoms_channel, batch_first=True, padding_value=PADDING_INDEX
        )
        radius = [item["receptor"]["radius"] for item in batch]
        radius = torch.nn.utils.rnn.pad_sequence(
            radius, batch_first=True, padding_value=PADDING_INDEX
        )
        center_coords = [item["receptor"]["center_coords"] for item in batch]
        center_coords = torch.stack(center_coords)

        receptor = {
            "id": id_,
            "atoms_channel": atoms_channel,
            "coords": coords,
            "radius": radius,
            "center_coords": center_coords,
            "data_type": default_collate([item["receptor"]["data_type"] for item in batch])
        }
        if "xs" in batch[0]["receptor"]:
            receptor.update({
                "xs": default_collate([item["receptor"]["xs"] for item in batch]),
            })

    id_ = default_collate([item["ligand"]["id"] for item in batch])
    coords = [item["ligand"]["coords"] for item in batch]
    coords = torch.nn.utils.rnn.pad_sequence(
        coords, batch_first=True, padding_value=PADDING_INDEX
    )
    atoms_channel = [item["ligand"]["atoms_channel"] for item in batch]
    atoms_channel = torch.nn.utils.rnn.pad_sequence(
        atoms_channel, batch_first=True, padding_value=PADDING_INDEX
    )
    radius = [item["ligand"]["radius"] for item in batch]
    radius = torch.nn.utils.rnn.pad_sequence(
        radius, batch_first=True, padding_value=PADDING_INDEX
    )
    ligand = {
        "id": id_,
        "atoms_channel": atoms_channel,
        "coords": coords,
        "radius": radius,
        "data_type": default_collate([item["ligand"]["data_type"] for item in batch])
    }
    if "xs" in batch[0]["ligand"]:
        ligand.update({
            "xs": default_collate([item["ligand"]["xs"] for item in batch]),
        })
    if "cdr_h3_seq" in batch[0]["ligand"]:
        ligand.update({
            "cdr_h3_seq": default_collate([item["ligand"]["cdr_h3_seq"] for item in batch]),
        })

    return {"receptor": receptor if has_receptor else None, "ligand": ligand}
