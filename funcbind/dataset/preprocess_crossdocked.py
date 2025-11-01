import argparse
import os
from pyuul import utils
import torch
from tqdm import tqdm

from funcbind.utils.constants import ELEMENT_2_HASH_FUNCBIND


def preprocess_sdfs(data_dir: str, fname="crossdocked_pocket10") -> None:
    """
    Preprocesses the SDF files in the specified data directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        None
    """
    data_dir = os.path.join(data_dir, fname)
    for pocket in tqdm(os.listdir(data_dir)):
        path = os.path.join(data_dir, pocket)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith("sdf"):
                    file = os.path.join(path, file)
                    with open(file, "r") as f:
                        lines = f.read().splitlines()
                        if lines[-1] != "$$$$":
                            with open(file, "a") as f:
                                f.write("$$$$")


def has_structure(data_dir: str, poc: str, fname="crossdocked_pocket10") -> bool:
    """
    Check if a structure exists in the given data directory.

    Args:
        data_dir (str): The directory containing the structure data.
        poc (str): The name of the structure file.

    Returns:
        bool: True if the structure exists, False otherwise.
    """
    target_pdb = os.path.join(data_dir, fname, poc)
    coords, _ = utils.parsePDB(target_pdb)
    if coords[0].shape[0] == 0:
        return False
    return True


def preprocess_split(data: dict, data_dir: str, fname="crossdocked_pocket10", rep_type='all_atom') -> list:
    """
    Preprocesses the data by parsing PDB and SDF files to extract information about pockets and ligands.

    Args:
        data (dict): A dictionary containing pairs of pocket and ligand IDs.
        data_dir (str): The directory path where the PDB and SDF files are located.

    Returns:
        list: A list of tuples, where each tuple contains the processed pocket and ligand data.
    """
    processed_data = []

    for i, (poc, lig) in tqdm(enumerate(data)):
        print(f">> processing sample {i + 1}/{len(data)}, pocket: {poc}, ligand: {lig}")

        # skip samples where structure of pocket is 0
        # a tiny amount of samples
        if not has_structure(data_dir, poc):
            print(">> ignore pocket id:", poc)
            continue

        # pocket
        pocket_pdb = os.path.join(data_dir, fname, poc)
        coords, atname = utils.parsePDB(pocket_pdb)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND)
        pocket = {
            "id": poc,
            "coords": coords[0].clone(),
            "atoms_channel": atoms_channel[0].type(torch.uint8),
        }

        # ligand
        ligand_sdf = os.path.join(data_dir, fname, lig)
        coords, atname = utils.parseSDF(ligand_sdf)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND)
        c_min, _ = coords[0].min(axis=0)
        c_max, _ = coords[0].max(axis=0)
        ligand = {
            "id": lig,
            "coords": coords[0].clone(),
            "atoms_channel": atoms_channel[0].type(torch.uint8),
            "max_len": round((c_max - c_min).max().item(), 2),
        }

        processed_data.append((pocket, ligand))
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--fname", type=str, default="crossdocked_pocket10")
    parser.add_argument("--rep_type", type=str, default="all_atom")
    args = parser.parse_args()

    preprocess_sdfs(args.data_dir, fname=args.fname)

    dir = os.path.join(args.data_dir, args.fname)
    split_by_name = torch.load(os.path.join(dir, "split_by_name.pt"))
    for split in ["train", "test"]:
        print(">> preprocessing split", split)
        data = preprocess_split(split_by_name[split], args.data_dir, fname=args.fname, rep_type=args.rep_type)
        torch.save(data, os.path.join(dir, f"data_{split}.pt"))