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
        assert split in ["train", "val", "test"]
        assert "use_single_dataset" not in config["dset"]  or config["dset"]["use_single_dataset"] in [None, "xdocked", "mcpp", "sabdab"]
        assert "datasets" not in config["dset"] or config["dset"]["datasets"] is None or set(config["dset"]["datasets"]) <= {"xdocked", "mcpp", "sabdab"}

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

        # Which sub-datasets to concatenate. `datasets` is the explicit form and
        # wins when given; `use_single_dataset` stays supported so existing runs
        # keep behaving identically (None meaning "every sub-dataset").
        requested = config["dset"].get("datasets", None) if hasattr(config["dset"], "get") else None
        if requested:
            self.datasets = set(requested)
        elif self.use_single_dataset is not None:
            self.datasets = {self.use_single_dataset}
        else:
            self.datasets = {"xdocked", "mcpp", "sabdab"}

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
        self.xdocked_source_index = []
        if "xdocked" in self.datasets:
            _xdocked = DatasetReceptorLigand(
                data_dir=config["dset"]["data_dir"],
                input_dataset="crossdocked_pocket10",
                split=split,
                val_from_train_pool=bool(config["dset"].get("val_from_train_pool", False)),
            )
            data_xdocked = _xdocked.data
            # Position within range_xdocked -> density-crop id. Needed because the crops
            # are named by their row in the shuffled data_train.pt, not by omni index.
            self.xdocked_source_index = _xdocked.source_index
            self.data.extend(data_xdocked)

        # load mcpp
        if "mcpp" in self.datasets:
            data_mcpp = DatasetMCPP(
                data_dir=config["dset"]["data_dir"],
                input_dataset="mcpp_dataset",
                split=split,
                elements=config["dset"]["elements"],
            ).data
            self.data.extend(data_mcpp)

        # load sabdab
        if "sabdab" in self.datasets:
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

        # ---- experimental density (off unless dset.density_crops_dir is set) --------
        # Crops are named by their row in the shuffled data_train.pt, which is exactly
        # xdocked_source_index[i]; see dataset_crossdocked for why the two line up.
        self.density_crops_dir = config["dset"].get("density_crops_dir", "") or ""
        self.voxbind_root = config["dset"].get("voxbind_python_root") or os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind")
        self.density_available = None
        if self.density_crops_dir:
            import numpy as np
            phys = "test" if split == "test" else "train"
            self.density_crops_dir = os.path.join(self.density_crops_dir, phys)
            mask = os.path.join(os.path.dirname(self.density_crops_dir), f"{phys}_available.npy")
            if os.path.exists(mask):
                self.density_available = np.load(mask)
            n_av = int(self.density_available.sum()) if self.density_available is not None else -1
            print(f"density crops: {self.density_crops_dir} (available={n_av:,}, split={split})")

        # MCP uses one larger, raw holo-density box per deposited receptor. The
        # current conformer centre and atom augmentation are applied at read time.
        self.mcpp_holo_density_dir = (
            config["dset"].get("mcpp_holo_density_dir", "") or ""
        )
        self.mcpp_holo_density = None
        if self.mcpp_holo_density_dir and "mcpp" in self.datasets:
            from funcbind.dataset.mcpp_holo_density import MCPPHoloDensityStore
            self.mcpp_holo_density = MCPPHoloDensityStore(self.mcpp_holo_density_dir)
            n_av = sum(self.mcpp_holo_density.available(key)
                       for key in self.mcpp_holo_density.records)
            print(f"MCP holo density: {self.mcpp_holo_density_dir} "
                  f"(target maps={n_av:,}, split={split})")
        self.has_density = bool(self.density_crops_dir or self.mcpp_holo_density)

        self.range_xdocked = list(range(len(data_xdocked)))
        self.range_mcpp = list(range(len(data_xdocked), len(data_xdocked) + len(data_mcpp)))
        self.range_sabdab = list(range(len(data_xdocked)+len(data_mcpp), len(data_xdocked)+len(data_mcpp)+len(data_sabdab)))
        if self.rebalance:
            # Skip sub-datasets that were not loaded, otherwise their empty range
            # drives min() to zero and the epoch collapses to length 0.
            self.cluster_dict = [
                idxlist
                for idxlist in (self.range_xdocked, self.range_mcpp, self.range_sabdab)
                if len(idxlist) > 0
            ]


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

    def _load_density(self, index, rot_matrix, trans_noise, sample_center) -> dict:
        """Load receptor density and carry it through the same atom augmentation."""
        blank = {"density": torch.zeros(64, 64, 64), "density_available": torch.tensor(False)}

        mcpp_start = len(self.range_xdocked)
        mcpp_stop = mcpp_start + len(self.range_mcpp)
        if self.mcpp_holo_density is not None and mcpp_start <= index < mcpp_stop:
            from funcbind.dataset.mcpp_holo_density import mcpp_target_id
            target_id = mcpp_target_id(self.data[index][2])
            density, available = self.mcpp_holo_density.load(
                target_id,
                sample_center=sample_center,
                rotation=rot_matrix,
                translation=trans_noise,
            )
            return {"density": density, "density_available": available}

        if not self.density_crops_dir or index >= len(self.xdocked_source_index):
            return blank
        src = self.xdocked_source_index[index]
        if self.density_available is not None and not bool(self.density_available[src]):
            return blank
        path = os.path.join(self.density_crops_dir, f"{src:06d}.npy")
        if not os.path.exists(path):
            return blank

        from funcbind.dataset.dataset_crossdocked import _voxbind_density_ops
        rotate_density, translate_density = _voxbind_density_ops(self.voxbind_root)

        d = np.load(path).astype(np.float32)
        if rot_matrix is not None:
            d = rotate_density(d, rot_matrix)
        if trans_noise is not None:
            d = translate_density(d, trans_noise)
        return {"density": torch.from_numpy(d), "density_available": torch.tensor(True)}

    def __getitem__(self, index) -> dict:
        ligand, receptor = self.get_ligand_receptor(index)

        sample_center = receptor["center_coords"].clone()
        # center reference of frame to center of mass of ligand
        ligand, receptor = recenter_structures(ligand, receptor, receptor["center_coords"])

        # add aug (rotation then translation)
        rot_matrix = trans_noise = None
        if self.aug and self.has_density:
            # Same augmentation, but the transform is captured so the density volume can
            # follow the atoms. Translation is inlined because translate_coords does not
            # return its noise vector.
            from funcbind.utils.utils_sampling import random_rot_matrix
            rot_matrix = random_rot_matrix()
            ligand, receptor = rotate_coords(ligand, receptor, rot_matrix)
            trans_noise = (torch.rand((1, 3), dtype=ligand["coords"].dtype) - 0.5) * 2 * self.delta
            ligand["coords"] = ligand["coords"] + trans_noise
            if receptor is not None:
                receptor["coords"] = receptor["coords"] + trans_noise
        elif self.aug:
            ligand, receptor = rotate_coords(ligand, receptor)
            ligand, receptor = translate_coords(ligand, receptor, self.delta)

        # box molecules
        ligand = filter_atoms_by_distance_mask(ligand, max_dim=self.max_dim)
        if receptor is not None:
            receptor = filter_atoms_by_distance_mask(receptor, max_dim=self.max_dim)

        # add xs
        if self.sample_points:
            ligand.update({"xs": self._get_xs(ligand)})

        out = {"receptor": receptor, "ligand": ligand}
        if self.has_density:
            out.update(self._load_density(
                index, rot_matrix, trans_noise, sample_center
            ))
        return out


def _max_abs_coord(coords):
    return max(
        (coord.abs().max().item() for coord in coords if coord.numel() > 0),
        default=0.0,
    )


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
            "max_abs_coord": _max_abs_coord(
                [item["receptor"]["coords"] for item in batch]
            ),
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
        "max_abs_coord": _max_abs_coord(
            [item["ligand"]["coords"] for item in batch]
        ),
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

    out = {"receptor": receptor if has_receptor else None, "ligand": ligand}
    if "density" in batch[0]:
        out["density"] = torch.stack([item["density"] for item in batch])
        out["density_available"] = torch.stack([item["density_available"] for item in batch])
    return out
