from typing import Dict, Optional
import argparse
import os
import enum
import torch
from tqdm.contrib.concurrent import process_map
from cpdb import parse
import pandas as pd
from pandas import DataFrame
from biopandas.pdb.engines import amino3to1dict

from funcbind.utils.constants import AB_CHAIN_DICT, ELEMENT_2_HASH_FUNCBIND, PEPTIDE_ALPHABET
from anarci import run_anarci

# CDR region selection is taken from https://github.com/luost26/diffab/blob/main/diffab/datasets/sabdab.py
class CDR(enum.IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    L1 = 4
    L2 = 5
    L3 = 6
    Fv = 0

def str_to_cdr(str):
    if str == "H1":
        cdr = CDR.H1
    elif str == "H2":
        cdr = CDR.H2
    elif str == "H3":
        cdr = CDR.H3
    elif str == "L1":
        cdr = CDR.L1
    elif str == "L2":
        cdr = CDR.L2
    elif str == "L3":
        cdr = CDR.L3
    else:
        cdr = CDR.Fv
    return cdr


class ChothiaCDRRange:
    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    @classmethod
    def to_cdr(cls, chain_type, resseq):
        assert chain_type in ('H', 'L')
        if chain_type == 'H':
            if cls.H1[0] <= resseq <= cls.H1[1]:
                return CDR.H1
            elif cls.H2[0] <= resseq <= cls.H2[1]:
                return CDR.H2
            elif cls.H3[0] <= resseq <= cls.H3[1]:
                return CDR.H3
            else:
                return CDR.Fv
        elif chain_type == 'L':
            if cls.L1[0] <= resseq <= cls.L1[1]:     # Chothia VH-CDR1
                return CDR.L1
            elif cls.L2[0] <= resseq <= cls.L2[1]:
                return CDR.L2
            elif cls.L3[0] <= resseq <= cls.L3[1]:
                return CDR.L3
            else:
                return CDR.Fv
        else:
            return CDR.Fv


def aa_tensor_to_sequence(aa_tensor):
    return ''.join([PEPTIDE_ALPHABET[aa.item()] for aa in aa_tensor.flatten()])

def get_numbering(sequence, scheme='chothia'):
    results = run_anarci(seq=sequence, scheme=scheme)
    if results is None:
        return None
    numbering = [num[0] for num, aa in results[1][0][0][0] if aa.strip() != '-']
    return torch.tensor(numbering, dtype=torch.long)

def process_file(file_info: Dict[str, str]) -> None:
    ab_path, ag_path, out_path = file_info

    # Load the PDB files
    ab_df = parse(ab_path, df=True)

    # Drop hydrogens
    ab_df = ab_df[ab_df["element_symbol"] != "H"]

    # Get the sequences
    ab_seq = ab_df.groupby(["chain_id", "residue_number"]).first()["residue_name"].values
    ab_seq = [amino3to1dict.get(s.strip(), 'X') for s in ab_seq]
    ab_seq = torch.tensor([PEPTIDE_ALPHABET.index(s) for s in ab_seq], dtype=torch.long)

    # Chain IDs as integers
    ab_chain_ids = torch.tensor([AB_CHAIN_DICT[c] for c in ab_df.groupby(["chain_id", "residue_number"], as_index=False).first()["chain_id"].values])

    # Get the CDR regions
    cdr_region = torch.zeros_like(ab_chain_ids)
    for chain_idx, chain_str_id in enumerate(['H', 'L']):
        if len(ab_seq[ab_chain_ids == chain_idx]) > 0:
            mask = (ab_chain_ids == chain_idx)
            numbering = get_numbering(aa_tensor_to_sequence(ab_seq[mask]), scheme="chothia")
            numbering = torch.tensor([ChothiaCDRRange.to_cdr(chain_str_id, chothia_idx) for chothia_idx in numbering], dtype=torch.long)
            # If numbering is shorter than the sequence, pad it with Fv (can happen if ANARCI drops trailing residues)
            if len(numbering) < len(ab_seq[mask]):
                numbering = torch.cat([numbering, torch.full((len(ab_seq[mask]) - len(numbering),), CDR.Fv, dtype=torch.long)])
            cdr_region[mask] = numbering

    # Get the coordinates
    ab_coords = torch.from_numpy(
        ab_df[["x_coord", "y_coord", "z_coord"]].values
    ).float()

    # Get encoding of atom types as integers
    ab_atom_types = torch.from_numpy(
        ab_df["element_symbol"].map(ELEMENT_2_HASH_FUNCBIND).values
    ).long()

    # Get atom chain IDs as integers
    ab_atom_chain_ids = torch.from_numpy(ab_df["chain_id"].map(AB_CHAIN_DICT).values).long()

    # Get atom residue IDs as integers
    ab_atom_residue_ids = torch.tensor(ab_df["residue_number"].values).long()
    # Ensure that residue IDs are contiguous
    offset = 0
    for chain_id in ab_atom_chain_ids.unique():
        chain_mask = (ab_atom_chain_ids == chain_id)
        unique_residue_ids = ab_atom_residue_ids[chain_mask].unique()
        for i, residue_id in enumerate(unique_residue_ids):
            ab_atom_residue_ids[chain_mask & (ab_atom_residue_ids == residue_id)] = i + offset
        offset += len(unique_residue_ids)

    ab = {
        "id": ab_path.split("/")[-1].split(".")[0],
        "coords": ab_coords,
        "atoms_channel": ab_atom_types,
        "atom_chain_ids": ab_atom_chain_ids,
        "atom_residue_ids": ab_atom_residue_ids,
        "sequences": ab_seq,
        "sequence_chain_ids": ab_chain_ids,
        "cdr_region": cdr_region,
    }

    if ag_path is None:
        ag = None
    else:
        ag_df = parse(ag_path, df=True)

        # Drop hydrogens
        ag_df = ag_df[ag_df["element_symbol"] != "H"]

        # Get the sequences
        ag_seq = ag_df.groupby(["chain_id", "residue_number"]).first()["residue_name"].values
        ag_seq = [amino3to1dict.get(s.strip(), 'X') for s in ag_seq]
        ag_seq = torch.tensor([PEPTIDE_ALPHABET.index(s) for s in ag_seq], dtype=torch.long)

        # Chain IDs as integers
        ag_chain_ids = torch.tensor([AB_CHAIN_DICT[c] for c in ag_df.groupby(["chain_id", "residue_number"], as_index=False).first()["chain_id"].values])

        # Get the coordinates
        ag_coords = torch.from_numpy(
            ag_df[["x_coord", "y_coord", "z_coord"]].values
        ).float()

        # Get encoding of atom types as integers
        ag_atom_types = torch.from_numpy(
            ag_df["element_symbol"].map(ELEMENT_2_HASH_FUNCBIND).values
        ).long()

        # Get atom chain IDs as integers
        ag_atom_chain_ids = torch.from_numpy(ag_df["chain_id"].map(AB_CHAIN_DICT).values).long()

        # Get atom residue IDs as integers
        ag_atom_residue_ids = torch.from_numpy(ag_df["residue_number"].values).long() - 1

        ag = {
            "id": ag_path.split("/")[-1].split(".")[0],
            "coords": ag_coords,
            "atoms_channel": ag_atom_types,
            "atom_chain_ids": ag_atom_chain_ids,
            "atom_residue_ids": ag_atom_residue_ids,
            "sequences": ag_seq,
            "sequence_chain_ids": ag_chain_ids,
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((ag, ab), out_path)

def _split_sabdab_diffab(
    df: DataFrame,
    split_csv_path: str,
) -> tuple[DataFrame]:
    # Use the DiffAb split to compare to it and other baselines in CDR H3 prediction
    split_df = pd.read_csv(split_csv_path)
    split_df["id"] = split_df["id"].str.split("_").str[0]
    split_df = split_df.drop_duplicates(subset=["id"], keep="first")
    split_df["id"] = split_df["id"].str.lower()
    df["pdb_code"] = df["pdb_code"].str.lower()
    df["pdb_code"] = df["pdb_code"].str.split("_").str[0]

    df = df.merge(
        split_df, left_on="pdb_code", right_on="id", how="inner"
    )

    return df

def process_file_wrapper(args):
    return process_file(*args)

def preprocess_split(split: str, source_csv_path: str, source_pdb_path: str, data_dir: str, dname: str = "sabdab", max_workers: int = 8, split_csv_path: Optional[str] = None) -> list:
    """
    Preprocesses the data by parsing PDB and SDF files to extract information about pockets and ligands.

    Args:
        data (dict): A dictionary containing pairs of pocket and ligand IDs.
        data_dir (str): The directory path where the PDB and SDF files are located.

    Returns:
        list: A list of tuples, where each tuple contains the processed pocket and ligand data.
    """

    pdb_path = os.path.join(data_dir, dname, "raw_pdbs")
    os.makedirs(pdb_path, exist_ok=True)
    # Download the dataset if needed
    if not os.listdir(pdb_path):
        print(f"Downloading PDBs from {source_pdb_path}...")
        os.system(
            f'aws s3 cp {source_pdb_path} {os.path.join(data_dir, dname, "sabdab.tar.gz")}'
        )
        print("Extracting PDBs, this may take a while...")
        os.makedirs(pdb_path, exist_ok=True)
        os.system(
            f'tar -xf {os.path.join(data_dir, dname, "sabdab.tar.gz")} -C {pdb_path} --strip-components=1'
        )
        os.system(f'rm {os.path.join(data_dir, dname, "sabdab.tar.gz")}')

    # Load the CSV file listing Abs
    if source_csv_path.endswith(".parquet"):
        data_df = pd.read_parquet(source_csv_path)
    else:
        data_df = pd.read_csv(source_csv_path)

    # Split the data into train, val, and test
    data_df = _split_sabdab_diffab(data_df, split_csv_path)
    if split == 'test':
        # For diffab test split, we only keep complex structures
        data_df = data_df[~data_df["ag_fname"].isnull()]
    data_df = data_df[data_df["split"] == split]

    out_dir = os.path.join(data_dir, dname, split)
    os.makedirs(out_dir, exist_ok=True)

    # Make paths to the PDB files
    data_df["ab_path"] = data_df["ab_fname"].apply(lambda x: os.path.join(pdb_path, x))
    no_ag_mask = data_df["ag_fname"].isnull()
    data_df["ag_path"] = None
    data_df.loc[~no_ag_mask, "ag_path"] = data_df.loc[~no_ag_mask, "ag_fname"].apply(lambda x: os.path.join(pdb_path, x))
    data_df["out_path"] = data_df["ab_fname"].apply(lambda x: os.path.join(out_dir, x.replace(".pdb", ".pt")))
    data_df.loc[~no_ag_mask, "out_path"] = data_df.loc[~no_ag_mask, "complex_fname"].apply(lambda x: os.path.join(out_dir, x.replace(".pdb", ".pt")))
    # Keep only the columns we are interested in
    data_df = data_df[["ab_path", "ag_path", "out_path"]]

    file_info = [(row) for _, row in data_df.iterrows()]
    print(f"Number of samples in {split} split: {len(file_info)}")
    process_map(process_file_wrapper, file_info, desc="Processing files", max_workers=max_workers, chunksize=max(1, len(file_info)//max_workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--source_csv_path", type=str)  # sabdab_2024-11-08.parquet
    parser.add_argument("--source_pdb_path", type=str)  # sabdab.tar.gz
    parser.add_argument("--split_csv_path", type=str)  # diffab_sabdab_split.csv
    parser.add_argument("--dname", type=str, default="sabdab_v0.5.2")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    args.dname = args.dname + "_diffab_chothia"

    dir = os.path.join(args.data_dir, args.dname)
    for split in ["train", "val", "test"]:
        print(">> preprocessing split", split)
        preprocess_split(split, args.source_csv_path, args.source_pdb_path, args.data_dir, dname=args.dname, max_workers=args.max_workers, split_csv_path=args.split_csv_path)
