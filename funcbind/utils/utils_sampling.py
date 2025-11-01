from collections import Counter, defaultdict
import math
import os
from copy import deepcopy
from funcbind.models.encoder import sample_posterior
from funcbind.utils.constants import (
    is_antibody_dataset, N_RECEPTOR_ELEMENTS
)
from funcbind.utils.utils_convert import mol2rdkit_obabel
import shutil
from funcbind.utils.utils_metrics import extract_sequences_from_pdb
from funcbind.utils.utils_vis import convert_sdf_to_pdb
from funcbind.metrics.metrics_mcpp import MetricsSamplingMCPP
import torch
from rdkit import RDLogger
from rdkit import Chem
import numpy as np
import pandas as pd
import random
from datetime import datetime
from cpdb import parse
from biopandas.pdb.engines import amino3to1dict, pdb_atomdict
from tqdm import tqdm

from funcbind.utils.utils_base import rotate_coords_single
from biopandas.pdb import PandasPdb

RDLogger.logger().setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog("rdApp.info")


def recenter_mols(mols: list, center_coords: torch.Tensor, clone=False) -> list:
    """
    Recenter the molecules based on the given center coordinates.

    Args:
        mols (list): List of molecules.
        center_coords (torch.Tensor): Center coordinates.

    Returns:
        list: List of recentered molecules.
    """
    if center_coords is not None:
        assert len(center_coords.shape) == 2, "Center coordinates must be a 2D tensor."
    centered_mols = []
    for i, mol in enumerate(mols):
        coords = mol["coords"]
        if clone:
            coords = coords.clone()
        if center_coords is not None:
            if center_coords.size(0) == 1:
                center_coords_ = center_coords.unsqueeze(0).tile(
                    (1, coords.shape[0], 1)
                )
            else:
                center_coords_ = (
                    center_coords[i].unsqueeze(0).tile((1, coords.shape[0], 1))
                )
            # Ensure same device for tensor addition
            center_coords_ = center_coords_.to(coords.device)
            coords += center_coords_
        centered_mols.append(
            {
                "coords": coords,
                "atoms_channel": mol["atoms_channel"],
                "radius": mol["radius"],
            }
        )

    return centered_mols


def make_batch_single(ligand_gt: dict, rotation=None, n_chains=None) -> dict:
    """Create a batch of ligands and receptors.

    Args:
        ligand_gt (dict): The ground truth ligand dictionary.
        receptor (dict): The receptor dictionary.
        n_samples (int): The number of samples to create in the batch.
        rotate (bool, optional): Whether to rotate the coordinates. Defaults to False.

    Returns:
        dict: A dictionary containing the ligands and receptors in the batch.
    """
    ligands = []
    if rotation is None:
        n_samples = n_chains
    else:
        n_samples = len(rotation)
    for i in range(n_samples):
        ligand_ = deepcopy(ligand_gt)
        if rotation is not None:
            ligand_ = rotate_coords_single(ligand_, rotation[i].to(ligand_["coords"].device))
            # need to unsqueeze after rotation with rotate_coords
            ligand_["coords"] = ligand_["coords"].unsqueeze(0)
        ligands.append(ligand_)
    data_types = torch.concat([lig["data_type"] for lig in ligands]).to("cpu") if "data_type" in ligand_gt else None
    ligands = {
        "coords": torch.concat([lig["coords"] for lig in ligands]).to("cpu"),
        "atoms_channel": torch.concat([lig["atoms_channel"] for lig in ligands]).to("cpu"),
        "radius": torch.concat([lig["radius"] for lig in ligands]).to("cpu"),
    }
    if data_types is not None:
        ligands.update({"data_type": data_types})
    return ligands


def make_batch_rotated_complexes(ligand_gt, receptor, rand_rots, fabric):
    # rotate both complexes
    batch = {"receptor": make_batch_single(receptor, rand_rots), "ligand": make_batch_single(ligand_gt, rand_rots)}
    # convert all subkeys of batch to cuda tensors
    for key in batch:
        if isinstance(batch[key], dict):
            for subkey in batch[key]:
                batch[key][subkey] = batch[key][subkey].to(fabric.device)
        else:
            batch[key] = batch[key].to(fabric.device)
    return batch


def path_to_raw_pdb(config):
    input_dataset = config["dset"]["input_dataset"]
    if config.get("pdb_path", None):
        return ["test_raw_pdbs"]
    elif (input_dataset == "omni_v1" and config["dset"]["use_single_dataset"] == "xdocked"):
        folders = ["crossdocked_pocket10"]
    elif (input_dataset == "omni_v1" and config["dset"]["use_single_dataset"] == "sabdab"):
        folders = ["sabdab_v0.5.2_diffab_chothia/raw_pdbs"]
    elif (input_dataset == "omni_v1" and config["dset"]["use_single_dataset"] == "mcpp"):
        folders = ["mcpp_dataset"]
    else:
        raise ValueError(f"Invalid input dataset: {input_dataset}")
    return folders


def render_samples(xhat, dec_module, fabric, config, rand_rots=None, batch_size_render_codes=100, unnormalize=True, target_dirname=None):
    if xhat.size(0) < batch_size_render_codes:
        batch_size_render_codes = xhat.size(0)
    batched_codes = torch.split(xhat, batch_size_render_codes, dim=0)
    mols = []

    fabric.print(
        f">> Splitting codes for rendering in batches of {batch_size_render_codes}"
    )
    if rand_rots is not None:
        rand_rots_tensor = torch.stack(rand_rots, dim=0)
        batched_rotations = torch.split(
            rand_rots_tensor, batch_size_render_codes, dim=0
        )
        for batched_code, batched_rot in tqdm(zip(batched_codes, batched_rotations)):
            mols += dec_module.codes_to_molecules_batched(
                batched_code.to(fabric.device),
                unnormalize=unnormalize,
                fabric=fabric,
                config=config,
                rand_rots=batched_rot.to(fabric.device),
                verbose=True,
                target_dirname=target_dirname,
            )
    else:
        for batched_code in tqdm(batched_codes):
            mols += dec_module.codes_to_molecules_batched(
                batched_code.to(fabric.device),
                unnormalize=unnormalize,
                fabric=fabric,
                config=config,
                verbose=True,
                target_dirname=target_dirname,
            )
    return mols


def save_samples(rdkmols, target_dirname, receptor, ligand_gt, config, fabric, seq_seed=None):
    if (ligand_gt is not None and "id" in ligand_gt) or ("id" in receptor):
        save_receptor_and_ligand(
            ligand_gt,
            receptor,
            target_dirname,
            config=config,
        )
    seq_gen = save_sdf_pdb(
        rdkmols,
        target_dirname,
        fabric,
        n_mols=config["sampling"]["n_samples_per_receptor"],
        to_pdb=is_antibody_dataset(config),
        ligand_id=ligand_gt["id"][0] if ligand_gt is not None and "id" in ligand_gt else None
    )
    if seq_seed is not None:
        fabric.print("ground truth seq", seq_seed)
    if (ligand_gt is not None and "id" in ligand_gt) or ("id" in receptor):
        try:
            save_full_ligand(
                "samples",
                seq_seed,
                ligand_gt,
                target_dirname,
                config=config,
                receptor=receptor,
            )
        except Exception as e:
            fabric.print(f"Error saving full ligand: {e}")

    return seq_gen


def normalize_code(codes, code_stats):
    """
    Normalize the given codes using standardization.

    Args:
        codes (torch.Tensor): The codes to be normalized.
        mean (torch.Tensor): The mean values for each code dimension.
        std (torch.Tensor): The standard deviation values for each code dimension.
    """
    mean = code_stats["mean"]
    std = code_stats["std"]
    return (codes - mean) / std


def save_pocket_and_ligand(pdb_file: str, sdf_file: str, out_dir: str) -> None:
    """
    Save the ligand and pocket files to the specified directory.

    Args:
        pdb_file (str): Path to PDB file (local or S3 URL)
        sdf_file (str): Path to SDF file (local or S3 URL)
        out_dir (str): Output directory

    Returns:
        None
    """
    # Handle pdb_file
    if pdb_file.startswith('s3://'):
        os.system(f"aws s3 cp {pdb_file} {out_dir}/")
    else:
        shutil.copyfile(
            pdb_file,
            os.path.join(out_dir, os.path.basename(pdb_file))
        )

    # Handle sdf_file
    if sdf_file is not None:
        if sdf_file.startswith('s3://'):
            os.system(f"aws s3 cp {sdf_file} {out_dir}/")
        else:
            shutil.copyfile(
                sdf_file,
                os.path.join(out_dir, os.path.basename(sdf_file))
            )


def save_receptor_and_ligand(
    ligand_gt,
    receptor,
    dirname,
    config,
) -> None:
    """
    Save the ligand and receptor files to the specified directory.

    Args:
        ligand_gt (dict): The ligand ground truth.
        receptor (dict): The receptor information.
        data_dir (str): The directory containing the data files.
        dirname (str): The directory to save the ligand and receptor files.

    Returns:
        None
    """
    folders = path_to_raw_pdb(config)
    data_dir = config["dset"]["data_dir"]
    ligand_id, receptor_id = ligand_gt["id"][0] if ligand_gt is not None else None, receptor["id"][0] if receptor is not None else None
    if is_antibody_dataset(config):
        ligand_id = f"{ligand_id}.pdb" if ligand_id is not None else None
        receptor_id = f"{receptor_id}.pdb" if receptor_id is not None else None

    for folder in folders:
        # Only try to copy ligand file if ligand_id is not None
        if ligand_id is not None:
            try:
                shutil.copyfile(
                    os.path.join(data_dir, folder, ligand_id),
                    os.path.join(dirname, ligand_id.replace("/", "__")),
                )
            except FileNotFoundError:
                print(f"File not found: {os.path.join(data_dir, folder, ligand_id)}")
                pass
        # Only try to copy receptor file if receptor_id is not None
        if receptor_id is not None:
            try:
                shutil.copyfile(
                    os.path.join(data_dir, folder, receptor_id),
                    os.path.join(dirname, receptor_id.replace("/", "__")),
                )
            except FileNotFoundError:
                print(f"File not found: {os.path.join(data_dir, folder, receptor_id)}")
                pass


def save_full_ligand(
    fname,
    seq_seed,
    ligand_gt,
    dirname,
    config,
    receptor=None,
) -> None:
    """
    For each generated ligand, in the ground truth structure, replace the original CDR H3 loop with the generated one.

    Args:
        ligand_pred_pdb_path (str): The path to the generated ligand pdb file.
        seq_seed (str): The ground truth CDR H3 loop sequence.
        ligand_gt (dict): The ground truth ligand information.
        dirname (str): The directory to save the ligand files.
    """
    ligand_id = ligand_gt["id"][0] if ligand_gt is not None else None
    if is_antibody_dataset(config):
        ligand_id = f"{ligand_id}.pdb"
    else:
        return None
    ligand_gt_path = os.path.join(dirname, ligand_id.replace("/", "__"))
    ligand_gt_df = parse(ligand_gt_path, df=True)
    ligand_gt_df = ligand_gt_df[ligand_gt_df["element_symbol"] != "H"]

    # Find the residue ids corresponding to seq_seed
    seq_seed = seq_seed.replace(" ", "")
    ligand_residue_df = ligand_gt_df.groupby(["chain_id", "residue_number"]).first()
    ligand_full_seq = "".join(
        [
            amino3to1dict.get(s.strip(), "X")
            for s in ligand_residue_df["residue_name"].values
        ]
    )
    start_id = ligand_full_seq.find(seq_seed)
    end_id = start_id + len(seq_seed)
    mask_residue_ids = ligand_residue_df.index.get_level_values(
        "residue_number"
    ).values[start_id:end_id]
    mask_chain_id = ligand_residue_df.index.get_level_values("chain_id").values[
        start_id
    ]

    # Split the ligand into two parts: before and after the CDR H3 loop
    ligand_gt_df = ligand_gt_df.sort_values(
        by=["model_idx", "chain_id", "residue_number"], na_position="first"
    )
    mask = (ligand_gt_df["chain_id"] == mask_chain_id) & (
        ligand_gt_df["residue_number"].isin(mask_residue_ids)
    )
    if mask.sum() == 0:
        return None
    first_true_index = mask.idxmax()
    ligand_gt_df_pre = ligand_gt_df.loc[: first_true_index - 1].copy()
    last_true_index = mask[::-1].idxmax()
    ligand_gt_df_post = ligand_gt_df.loc[last_true_index + 1 :].copy()

    ligand_pred_path = os.path.join(dirname, f"{fname}.pdb")
    ligand_pred_df = parse(ligand_pred_path, df=True)
    ligand_pred_df = ligand_pred_df.sort_values(
        by=["model_idx", "chain_id", "residue_number"], na_position="first"
    )

    if receptor is not None:
        # We will optionally save the complex structure with the full ligand
        receptor_id = receptor["id"][0]
        if is_antibody_dataset(config):
            receptor_id = f"{receptor_id}.pdb"
        else:
            return None
        receptor_path = os.path.join(dirname, receptor_id.replace("/", "__"))
        receptor_df = parse(receptor_path, df=True)
        receptor_df = receptor_df[receptor_df["element_symbol"] != "H"]

    for model_idx in ligand_pred_df["model_idx"].unique():
        chains = ligand_pred_df[ligand_pred_df["model_idx"] == model_idx][
            "chain_id"
        ].unique()
        ligand_pred = None
        if len(chains) == 1:
            ligand_pred = ligand_pred_df[
                (ligand_pred_df["model_idx"] == model_idx)
            ].copy()
        if ligand_pred is None:
            continue
        else:
            # Fix chain id for insertion
            ligand_pred["chain_id"] = ligand_gt_df_pre["chain_id"].values[-1]
            # Fix sequence order for insertion
            start_pos = ligand_pred[ligand_pred["atom_name"] == "CA"][
                ["x_coord", "y_coord", "z_coord"]
            ].values[0]
            end_pos = ligand_pred[ligand_pred["atom_name"] == "CA"][
                ["x_coord", "y_coord", "z_coord"]
            ].values[-1]
            fv_end_pos = ligand_gt_df_pre[ligand_gt_df_pre["atom_name"] == "CA"][
                ["x_coord", "y_coord", "z_coord"]
            ].values[-1]
            if np.linalg.norm(start_pos - fv_end_pos) > np.linalg.norm(
                end_pos - fv_end_pos
            ):
                # We flip the sequence order such that the first atom is closer to the insertion point
                ligand_pred = ligand_pred.iloc[::-1]
            ligand_pred["residue_number"] += ligand_gt_df_pre["residue_number"].values[
                -1
            ]
            ligand_gt_df_post["residue_number"] += (
                ligand_pred["residue_number"].values[-1]
                - ligand_gt_df_post["residue_number"].values[0]
                + 1
            )
            # Concatenate the ligand parts
            ligand_pred_df_full = pd.concat(
                [ligand_gt_df_pre, ligand_pred, ligand_gt_df_post]
            )
            # Fix L chain residue indexing to start from 1 (if H shortens this would have shifted)
            try:
                L_mask = ligand_pred_df_full["chain_id"] == "L"
                ligand_pred_df_full.loc[L_mask, "residue_number"] = (
                    ligand_pred_df_full.loc[L_mask, "residue_number"]
                    - ligand_pred_df_full.loc[L_mask, "residue_number"].values[0]
                    + 1
                )
            except Exception as e:  # no L chain
                pass
            # Fix atom number after insertion
            ligand_pred_df_full["atom_number"] = np.arange(
                1, len(ligand_pred_df_full) + 1
            )

            # Write the new pdb file with full ligand
            df_to_pdb(
                ligand_pred_df_full,
                os.path.join(
                    dirname, f"{ligand_id.replace('.pdb', '')}_full_ab_{model_idx}.pdb"
                ),
            )

            # Optionally save the complex structure with the full ligand
            if receptor is not None:
                receptor_df["chain_id"] = "A"
                receptor_df["residue_number"] += (
                    1 - receptor_df["residue_number"].values[0]
                )
                ligand_pred_df_full = pd.concat([ligand_pred_df_full, receptor_df])
                ligand_pred_df_full["atom_number"] = np.arange(
                    1, len(ligand_pred_df_full) + 1
                )
                df_to_pdb(
                    ligand_pred_df_full,
                    os.path.join(
                        dirname,
                        f"{ligand_id.replace('.pdb', '')}_full_complex_{model_idx}.pdb",
                    ),
                )


def df_to_pdb(df: pd.DataFrame, fname: str) -> None:
    """Write a (cpdb) dataframe to a PDB file.

    Args:
        df (pd.DataFrame): The dataframe to write.
        fname (str): The name of the PDB file to write.
    """
    df = df.copy()
    # Add missing columns from pdb_atomdict
    for col in pdb_atomdict:
        if col["id"] not in df.columns:
            if col["type"] == str:
                df[col["id"]] = ""
            elif col["type"] == int:
                df[col["id"]] = 0
            elif col["type"] == float:
                df[col["id"]] = 0.0
    # Correct charge values
    df["charge"] = None
    # Drop columns not in pdb_atomdict
    df = df[[col["id"] for col in pdb_atomdict]]
    # Add line_idx
    df["line_idx"] = np.arange(1, len(df) + 1)
    # Write to PDB using biopandas
    ppdb = PandasPdb()
    ppdb._df = {"ATOM": df}
    ppdb.to_pdb(fname)


def filter_valid_mol(
    mols,
    center_coords=None,
    remove_fragment=False,
    valid_mols=[],
    valid_seq=[],
    attempts=0,
    fabric=None,
    is_ab=False,
    unique_only=True,
    verbose=True,
    len_seq=None,
    remove_atoms_too_close=False,
    config=None,
    target_dirname=None,
    df_mols=None,
    is_mcpp=False
):
    if center_coords is not None:
        mols = recenter_mols(mols, center_coords=center_coords)
    if not is_ab and len_seq is not None:
        len_seq = None
    failed_counter = 0
    df_mol = None
    if df_mols is None:
        df_mols = pd.DataFrame()
    for idx, mol in enumerate(mols):
        try:
            mol, ob_mol = mol2rdkit_obabel(mol, remove_fragment=remove_fragment, remove_atoms_too_close=remove_atoms_too_close, return_ob_mol=is_mcpp)
        except Exception as e:
            failed_counter += 1
            mol = None
        if mol is not None:
            if remove_fragment:
                if is_ab:
                    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    random_suffix = "-" + str(random.randint(0, 999999999999)).zfill(4)
                    timestamp = timestamp_prefix + random_suffix
                    os.makedirs(f"./temp_{timestamp}", exist_ok=True)
                    seq_gen = save_sdf_pdb([mol], f"./temp_{timestamp}", fabric, n_mols=None, to_pdb=True, verbose=False)
                    if len(seq_gen) > 0:
                        seq_gen = seq_gen[0]
                        if ("X" not in seq_gen):
                            if ((len_seq is not None and len(seq_gen) == len_seq) or len_seq is None) and ((seq_gen not in valid_seq and unique_only) or not unique_only):
                                valid_mols.append(mol)
                                valid_seq.append(seq_gen)
                    try:
                        os.system(f"rm -rf ./temp_{timestamp}")
                    except Exception as e:
                        fabric.print(f"Error removing temp folder: {e} pass")
                elif is_mcpp and config is not None:  # MCPP mode
                    num_mols = len(valid_mols)
                    df_mol = MetricsSamplingMCPP(config=config, target_dirname=target_dirname, mol=ob_mol, rdkmol= mol, valid_mols=num_mols).update(verbose=False)
                    if df_mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        df_mols = pd.concat([df_mols, df_mol], ignore_index=True)
                        valid_mols.append(mol)
                        valid_seq.append(smiles)
                else:
                    smiles = Chem.MolToSmiles(mol)
                    if (smiles not in valid_seq and unique_only) or not unique_only:
                        valid_mols.append(mol)
                        valid_seq.append(smiles)
            else:
                valid_mols.append(mol)
    if verbose:
        fabric.print(f"attempt {attempts}, n_valid_mol (remove_fragment: {remove_fragment}): {len(valid_mols)}, failed rdkit conversion: {failed_counter}")
    return valid_mols, valid_seq, df_mols


def select_population(
    population_mols,
    population_natoms,
    idx_mols,
    population_seq,
    valid_mols=[],
    valid_seq=[],
    fabric=None,
    verbose=True,
    len_seq=None,
    one_edit_away=False,
    top_natoms=True,
):
    selected_mols, selected_natoms, selected_idx_mols = [], [], []

    # Sequences within 1 edit from the seed
    if one_edit_away:
        idx_lens = [idx for idx, seq in population_seq.items() if abs(len(seq) - len_seq) <= 1]
        max_idx_lens = min(len(population_mols) // 4, len(idx_lens))
        for idx in idx_lens[:max_idx_lens]:
            mol_idx = idx_mols.index(idx)
            selected_mols.append(population_mols[mol_idx])
            selected_natoms.append(population_natoms[mol_idx])
            selected_idx_mols.append(idx)
    else:
        idx_lens = []

    # Biggest molecules in the population
    if top_natoms:
        sorted_pairs = sorted(zip(population_mols, population_natoms, idx_mols), key=lambda x: x[1], reverse=True)
        population_mols, population_natoms, idx_mols = [], [], []
        for mol, n_atoms, idx in sorted_pairs:
            if idx not in idx_lens:
                population_mols.append(mol)
                population_natoms.append(n_atoms)
                idx_mols.append(idx)
        top_count = max(len(population_mols) // 4 + 1 - len(idx_lens), 10)
        selected_mols.extend(population_mols[:top_count])
        selected_natoms.extend(population_natoms[:top_count])
        selected_idx_mols.extend(idx_mols[:top_count])
    else:
        top_count = 0
        assert one_edit_away, "If not top_natoms, one_edit_away must be True"

    if verbose:
        fabric.print(f"{len(idx_lens)} (1 edit) + {top_count} (largest natoms) / {len(population_mols)} = {(len(selected_natoms))/len(population_mols):.4f}")

        # Natoms in population and selected population
        fabric.print(f"{len(population_natoms)} population_mols: {sorted(Counter(population_natoms).items())}")
        fabric.print(f"{len(selected_natoms)} selected_population_mols: {sorted(Counter(selected_natoms).items())}")

        # Sequences in population and selected population
        counts = defaultdict(int)
        selected = []
        for seq in population_seq.values():
            l = len(seq)
            counts[l] += 1
            if l == len_seq:
                selected.append(seq)
        fabric.print(f"{len(population_seq)} seq_in_population: {sorted(counts.items())}, {selected}")
        seq_in_selected_population = [population_seq[idx] for idx in selected_idx_mols if idx in population_seq]
        fabric.print(f"{len(seq_in_selected_population)} seq_in_selected_population: {sorted(Counter([len(seq) for seq in seq_in_selected_population]).items())}, {seq_in_selected_population}")

        fabric.print(f"{len(valid_mols)} valid_mols of len {len_seq}, {valid_seq}")
    return selected_mols


def save_sdf_pdb(
    rdkmols,
    target_dirname,
    fabric,
    n_mols=None,
    fname="samples",
    to_pdb=False,
    verbose=True,
    ligand_id=None
):
    if n_mols is None:
        n_mols = len(rdkmols)
    if verbose:
        fabric.print(f"Saving in directory {target_dirname}, {min(n_mols, len(rdkmols))} molecules")

    # To SDF
    with Chem.SDWriter(os.path.join(target_dirname, f"{fname}.sdf")) as writer:
        for rdkmol in rdkmols[:n_mols]:
            try:
                writer.write(rdkmol)
            except Exception:
                fabric.print("cannot save", rdkmol)


    # SDF to PDB
    if to_pdb:
        convert_sdf_to_pdb(target_dirname, fabric=fabric, fname=f"{fname}.pdb", fname_sdf=f"{fname}.sdf")
        seq_gen = extract_sequences_from_pdb(pdb_file=os.path.join(target_dirname, f"{fname}.pdb"), target_dirname=target_dirname, ligand_id=ligand_id)
        if verbose:
            fabric.print(f"{len(seq_gen)} generated seq", seq_gen)
        return seq_gen
    return rdkmols


def infer_receptor_voxel(
    batch, field_maker, to_cpu=False, num_channels=None,
):
    voxels = field_maker.compute_voxel_grid(batch, num_channels=N_RECEPTOR_ELEMENTS if num_channels is None else num_channels)
    if to_cpu:
        voxels = voxels.cpu()
    return voxels


def batched_voxelization(molecules, field_maker, fabric, num_channels=None):
    voxelized_batches = []
    for start_idx in range(0, molecules["coords"].size(0), 500):
        end_idx = start_idx + 500
        batch_molecules = {
            k: v[start_idx:end_idx].to(fabric.device) for k, v in molecules.items()
        }
        voxel_batch = infer_receptor_voxel(batch_molecules, field_maker, to_cpu=True, num_channels=num_channels)
        voxelized_batches.append(voxel_batch)
    return torch.cat(voxelized_batches, dim=0)


def get_receptor_encoding(
    receptor, model, field_maker_receptor, fabric, rand_rots=None, n_chains=None
):
    receptors = make_batch_single(receptor, rand_rots, n_chains=n_chains)
    receptors_vox = batched_voxelization(
        receptors, field_maker_receptor, fabric=fabric
    )
    if not hasattr(model, "receptor_encoder") or model.receptor_encoder is None:
        return receptors_vox.to(fabric.device)
    return model.receptor_encoder(receptors_vox.to(fabric.device))


def get_ligand_encoding(
    ligand_gt, model, enc, field_maker, fabric, config_nf, rand_rots=None, n_chains=None
):
    ligands_gt = make_batch_single(ligand_gt, rand_rots, n_chains=n_chains)
    ligands_gt_vox = batched_voxelization(ligands_gt, field_maker, fabric=fabric, num_channels=len(config_nf["dset"]["elements"]))

    codes_ligand = enc(ligands_gt_vox.to(fabric.device))
    if config_nf["reg_weight"] != 0.0:
        codes_ligand = sample_posterior(codes_ligand)[0]
    return normalize_code(codes_ligand, model.code_stats)


def random_rot_matrix() -> torch.Tensor:
    """Apply random rotation in each of the x, y and z axis.
    First compute the 3D matrix for each rotation, then multiply them

    Returns:
        torch.Tensor: return rotation matrix (3x3)
    """
    theta = random.uniform(0, 2) * math.pi
    rot_x = torch.Tensor(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )
    theta = random.uniform(0, 2) * math.pi
    rot_y = torch.Tensor(
        [
            [math.cos(theta), 0, -math.sin(theta)],
            [0, 1, 0],
            [math.sin(theta), 0, math.cos(theta)],
        ]
    )
    theta = random.uniform(0, 2) * math.pi
    rot_z = torch.Tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    R = rot_z @ rot_y @ rot_x

    return R
