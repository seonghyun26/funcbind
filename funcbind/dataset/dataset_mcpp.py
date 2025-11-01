import os

import torch

from torch.utils.data import Dataset
from funcbind.utils.constants import ELEMENT_2_HASH_FUNCBIND


class DatasetMCPP(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/data/",
        input_dataset: str = "mcpp_dataset",
        split: str = "train",
        small: bool = False,
        elements: list = None,
    ):
        """
        Dataset class for MCPP data.
        The MCP elments are {'O', 'F', 'C', 'Br', 'S', 'Cl', 'N'}

        Args:
            data_dir (str, optional): Directory containing the dataset. Defaults to "dataset/data/".
            split (str, optional): Split of the dataset to use ("train", "val", or "test"). Defaults to "train".
            small (bool, optional): Whether to use a small subset of the dataset. Defaults to False.
            elements (list, optional): List of elements to filter by. Defaults to None.
        """
        assert split in ["train", "val", "test"]

        self.data_dir = os.path.join(data_dir, input_dataset)
        self.split = split
        self.small = small
        if elements is not None:
            elements = [ELEMENT_2_HASH_FUNCBIND[el] for el in elements]

        # Read data
        split = split if not small else "val"
        file = os.path.join(self.data_dir, f"{split}_data.pt")
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist. Please run the script to generate the dataset.")
        data = torch.load(file, weights_only=False)
        self.data = []
        for k,v in data.items():
            if len(v["prot_confs"]) == 0:
                continue
            for i in range(len(v["mcp_confs"])):
                if i >= len(v["prot_confs"]):
                    break
                if v["prot_confs"][i]['coords'].shape[0] == 0:
                    continue
                pair = [v["mcp_confs"][i], v["prot_confs"][i], v["mcp_paths"][i]]
                self.data.append(pair)

        # filter dataset
        # remove data where coords has shape 0
        self.data = [datum for datum in self.data if datum[0]["coords"].shape[0] != 0]
        self._filter_by_elements(elements=elements)
        self._filter_by_n_atoms(max_n_atoms=400)
        print("Loaded dataset", file, "with size", len(self.data))

    def _filter_by_n_atoms(self, max_n_atoms=200):
        filtered_data = []
        for datum in self.data:
            ligand = datum[0]
            n_atoms = ligand["coords"].shape[0]
            if n_atoms <= max_n_atoms:
                filtered_data.append(datum)
        if len(self.data) != len(filtered_data):
            print(
                f"  | filter data (n_atoms): data reduced from {len(self.data)} to {len(filtered_data)}"
            )
            self.data = filtered_data

    def _filter_by_elements(self, elements) -> None:
        if elements is None:
            return
        filtered_data = []
        for datum in self.data:
            ligand = datum[0]
            atoms = set(ligand["atoms_channel"])
            include = True
            for atom_id in atoms:
                if atom_id not in elements:
                    include = False
                    break
            if include:
                filtered_data.append(datum)
        if len(self.data) != len(filtered_data):
            print(
                f"  | filter data (elements): data reduced from {len(self.data)} to {len(filtered_data)}"
            )
            self.data = filtered_data
