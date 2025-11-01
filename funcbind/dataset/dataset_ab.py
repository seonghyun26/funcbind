import random
from funcbind.dataset.preprocess_sabdab import CDR, str_to_cdr
from torch.utils.data import Dataset
import os
import torch
from glob import glob
from tqdm import tqdm
from funcbind.utils.constants import PEPTIDE_ALPHABET
from funcbind.utils.utils_base import out_of_box, recenter_structures, filter_atoms_by_distance_mask, is_in_contact


class DatasetAntibodyAntigen(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/data",
        dataset_name: str = "sabdab_v0.5.2_diffab_chothia",
        split: str = "train",
        cdrs_aug = [],
        cdrs = ["H3"],
        max_aa_length: int = 20,
        grid_dim: int = 128,
        resolution: float = 0.25,
    ):
        """
        Dataset class for SAbDab, matches DiffAb splits.

        Args:
            data_dir (str, optional): Directory containing the dataset. Defaults to "dataset/data/".
            dataset_name (str, optional): Name of the dataset. Defaults to "sabdab_v0.5.2_diffab_chothia".
            split (str, optional): Split of the dataset to use ("train", "val", or "test"). Defaults to "train".
            cdrs_aug (list, optional): CDRs to use for augmentation. Defaults to [].
            cdrs (list, optional): CDRs to include. Defaults to ["H3"].
            rebalance (bool, optional): Whether to rebalance the dataset. Defaults to False.
            max_aa_length (int, optional): Maximum amino acid length for filtering. Defaults to 20.
            grid_dim (int, optional): Grid dimension for max_dim calculation. Defaults to 128.
            resolution (float, optional): Resolution for max_dim calculation. Defaults to 0.25.
        """
        assert split in ["train", "val", "test"]

        # data params
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.max_dim = (grid_dim * resolution) // 2  # Required by parent class filtering methods

        # ab params
        self.max_aa_length = max_aa_length
        self.cdrs_aug_str = cdrs_aug
        self.cdrs_aug = [str_to_cdr(cdr) for cdr in cdrs_aug]
        self.cdrs = [str_to_cdr(cdr) for cdr in cdrs]

        # Initialize file paths
        self._initialize_file_paths()

        # Load and filter dataset
        self.data = []
        self._load_datasets()

    def _initialize_file_paths(self):
        """Initialize file paths for different datasets."""
        # Sabdab filename
        self.sabdab_path = f"{self.data_dir}/{self.dataset_name}/{self.split}"
        self.sabdab_file = f"{self.sabdab_path}/filtered_{self.dataset_name}_data_{self.split}_mod.pt"

        # SabDab augmentation filename (non H3 loops)
        if len(self.cdrs_aug) == 5:
            self.sabdab_file_aug = f"{self.sabdab_path}/filtered_{self.dataset_name}_data_{self.split}_cdr_aug_all_mod.pt"
        elif len(self.cdrs_aug) == 1:
            self.sabdab_file_aug = f"{self.sabdab_path}/filtered_{self.dataset_name}_data_{self.split}_{self.cdrs_aug_str[0]}_mod.pt"
        else:
            self.sabdab_file_aug = None

    def _load_sabdab(self, path, file, cdrs=[CDR.H3]):
        size_init = len(self.data)
        if not os.path.exists(file):
            self.data += self._filter_sabdab(cdrs=cdrs, path=path, file=file)
        else:
            self.data += torch.load(file, weights_only=False)
        print("Loaded dataset", file, "with size", len(self.data) - size_init)

    # SabDab
    def _filter_sabdab(self, cdrs = [CDR.H3], path = None, file = None):
        """Filter out structures with no CDR H3."""
        filtered_data = []
        data_pt = list(glob(os.path.join(path, "**/*.pt"), recursive=True))
        print(f"Pre filtering, {len(data_pt)} structures.")
        for data in tqdm(data_pt):
            if "filtered_" not in data:
                try:
                    ag, ab = torch.load(data, weights_only=False)
                    should_append = True
                    for cdr in cdrs:
                        cdr_region_mask = (ab["cdr_region"] == cdr)
                        cdr_mask = cdr_region_mask[ab["atom_residue_ids"]]
                        if (cdr_mask.sum() == 0) or cdr_region_mask.sum() > 30:  # drop if no CDR or over 30 residues
                            should_append = False
                            break
                        if should_append:
                            recentered_ab, recentered_ag, recentered_cdr_region_mask = self._reduce_size_ab(ab, ag, cdr_mask, cdr_region_mask)
                            filtered_data.append((
                                None if ag is None else {"coords": recentered_ag["coords"], "atoms_channel": recentered_ag["atoms_channel"].to(torch.uint8), "id": recentered_ag["id"], "data_type": 2},
                                {"coords": recentered_ab["coords"], "atoms_channel": recentered_ab["atoms_channel"].to(torch.uint8), "id": recentered_ab["id"], "cdr_h3_seq": "".join(PEPTIDE_ALPHABET[idx] for idx in recentered_ab["sequences"][cdr_region_mask]), "cdr_mask": recentered_cdr_region_mask, "data_type": 2}
                            ))
                except Exception as e:
                    print(f"Error loading: {e}")
        if filtered_data:
            torch.save(filtered_data, file)
        print(f"After filtering, {len(filtered_data)} single loop structures across {len(cdrs)} CDRs saved to {file}.")
        return filtered_data


    def _reduce_size_ab(self, ab, ag, cdr_mask, cdr_region_mask):
        # Code to reduce size of the antibody and antigen to the sampled CDR region
        initial_coords = ab["coords"][cdr_mask].mean(0)
        recentered_ab, recentered_ag = recenter_structures(ab, ag, initial_coords)
        recentered_ab = filter_atoms_by_distance_mask(recentered_ab, max_dim=self.max_dim + 5)
        recentered_cdr_region_mask = cdr_region_mask[recentered_ab["atom_residue_ids"]]
        if ag is not None:
            recentered_ag = filter_atoms_by_distance_mask(recentered_ag, max_dim=self.max_dim + 5)
        recentered_ab, recentered_ag = recenter_structures(recentered_ab, recentered_ag, -initial_coords)  # in order to stitch back the sampled CDR and framework at sampling time
        return recentered_ab, recentered_ag, recentered_cdr_region_mask


    def _cdr_mask_to_fit_box(self, ab, ag, cdr_mask, cdr_region_mask):
        # Initialize the search by centering the CDR on the Ag
        cdr_coords = ab["coords"][cdr_mask]
        recentered_cdr, recentered_ag = recenter_structures({"coords": cdr_coords}, {"coords": ag["coords"]}, cdr_coords.mean(0))
        recentered_ag = filter_atoms_by_distance_mask(recentered_ag, max_dim=self.max_dim)
        if recentered_ag["coords"].shape[0] == 0:  # Skip if epitope not in box
            return None
        if not is_in_contact(recentered_ag, recentered_cdr, interface_threshold=10.0):  # Skip if not in contact
            return None

        # Trim CDR if it's out of box or too long
        max_allowed_size = self.max_dim - 2
        max_allowed_length = self.max_aa_length + 1
        while (out_of_box(recentered_cdr, max_dim=max_allowed_size) or cdr_region_mask.sum().item() > max_allowed_length):
            true_indices = torch.nonzero(cdr_region_mask, as_tuple=True)[0]
            if true_indices.numel() < 5:  # Stop if loop too short
                break
            # Remove first and last residue
            cdr_region_mask[true_indices[[0, -1]]] = False
            cdr_mask = cdr_region_mask[ab["atom_residue_ids"]]
            # Recalculate centered coordinates
            cdr_coords = ab["coords"][cdr_mask]
            recentered_cdr, _ = recenter_structures({"coords": cdr_coords}, None, cdr_coords.mean(dim=0))
        return cdr_mask, cdr_region_mask


    def _load_datasets(self):
        """Load all datasets based on configuration."""
        # Load SAbDab datasets
        assert len(self.cdrs) > 0 or len(self.cdrs_aug) > 0, "at least one CDR must be specified"
        if len(self.cdrs) > 0:
            self._load_sabdab(self.sabdab_path, self.sabdab_file, cdrs=self.cdrs)
        if len(self.cdrs_aug) > 0:
            self._load_sabdab(self.sabdab_path, self.sabdab_file_aug, cdrs=self.cdrs_aug)
