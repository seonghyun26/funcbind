import os

import numpy as np
from funcbind.metrics.metrics import MetricsSampling
from funcbind.metrics.metrics_ab import MetricsSamplingAb
from funcbind.metrics.metrics_mcpp import MetricsSamplingMCPP
from funcbind.utils.constants import PADDING_INDEX
from funcbind.metrics.metrics_crossdocked import MetricsSamplingCrossDocked
from cpdb import parse
from biopandas.pdb.engines import amino3to1dict
from rdkit import Chem
import yaml
import time


def create_sampling_metrics(config, target_dirname=None, df_mol=None):
    if config["dset"]["input_dataset"] == "crossdocked_pocket10" or (config["dset"]["input_dataset"] == "omni_v1" and config["dset"]["use_single_dataset"] == "xdocked"):
        return MetricsSamplingCrossDocked(config, target_dirname)
    elif (config["dset"]["input_dataset"] == "omni_v1" and config["dset"]["use_single_dataset"] == "sabdab") or ("diffab" in config["dset"]["input_dataset"]):
        return MetricsSamplingAb(config, target_dirname)
    elif "mcpp" in config["dset"]["input_dataset"] or (config["dset"]["input_dataset"] == "omni_v1" and config["dset"]["use_single_dataset"] == "mcpp"):
        return MetricsSamplingMCPP(config, target_dirname, df_mol)
    else:
        return MetricsSampling(config, target_dirname)


def log_results(
    metrics,
    t0: float,
    docking_mode: str,
    receptor_id: int = None,
    n_atoms_gt: int = None,
    fabric=None,
    target_dirname=None,
):
    """
    Logs the results of metrics sampling.

    Args:
        t0 (float): The starting time of the sampling process.
        logger (logging.Logger): The logger object used for logging.
        docking_mode (str): The docking mode.
        receptor_id (int, optional): The receptor ID. Defaults to None.
        n_atoms_gt (int, optional): The number of ground truth atoms. Defaults to None.
    """
    res_log, _ = metrics.log_results(fabric, t0, docking_mode, receptor_id, n_atoms_gt)
    if target_dirname is not None:
        with open(os.path.join(target_dirname, "metrics.yaml"), "w") as file:
            yaml.dump(convert_numpy_types(res_log), file)
    return res_log


# Convert NumPy data types to native Python data types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def evaluate_target_metrics(metrics_full, config, target_dirname=None, ligand_gt=None, receptor=None, seq_gen=None, seq_seed=None, attempts=None, receptor_id=0, t0=0, fabric=None, df_mol=None):
    t = time.time()
    metrics = create_sampling_metrics(config, target_dirname, df_mol=df_mol)
    metrics.update(ligand_gt=ligand_gt, receptor=receptor, seq_gen=seq_gen, seq_seed=seq_seed, attempts=attempts)
    metrics.save()

    metrics_full.molecular_stats.append(metrics.molecular_stats[0])
    if metrics_full.docking:
        metrics_full.res_dock.append(metrics.res_dock[0])
    if metrics_full.posecheck:
        metrics_full.res_posecheck.append(metrics.res_posecheck[0])
    if metrics_full.general:
        metrics_full.general_stats.append(metrics.general_stats[0])
    if metrics_full.segment_reconstruction:
        metrics_full.segment_reconstruction_scores.append(metrics.segment_reconstruction_scores[0])
    if metrics_full.mcpp_metrics:
        metrics_full.res_mcpp_metrics.append(metrics.res_mcpp_metrics[0])

    n_atoms_gt = (ligand_gt["atoms_channel"] != PADDING_INDEX).sum().item() if ligand_gt is not None else 0
    fabric.print(f"Time to sample molecules: {t - t0:.2f}s")
    fabric.print(f"Time to compute metrics: {time.time() - t:.2f}s")
    fabric.print(f"Time total: {time.time() - t0:.2f}s")
    log_results(metrics, t0, config["sampling"]["docking_mode"], receptor_id, n_atoms_gt, fabric=fabric)


def extract_sequences_from_pdb(pdb_file, seq_to_path=None, target_dirname=None, ligand_id=None):
    """
    Extracts amino acid sequences from a PDB file.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - list: A list of extracted sequences.
    """
    # Step 1: Parse PDB and extract necessary columns
    df = parse(pdb_file, df=True)[["atom_name", "residue_name", "chain_id", "model_idx", "residue_number"]]
    # Step 2: Identify residues with exactly one 'N', 'CA', and 'C' atom
    valid_residues = (df[df['atom_name'].isin(['N', 'CA', 'C'])].groupby(['model_idx', 'chain_id', 'residue_number'])['atom_name'].count().eq(3).reset_index(name='is_valid'))

    # Step 3: Determine which model_idx have all residues valid (original logic for single chain)
    fully_valid_models = valid_residues.groupby('model_idx')['is_valid'].all()
    valid_model_idxs = fully_valid_models[fully_valid_models].index
    # Step 4: Filter the DataFrame to include only fully valid model_idx
    df = df[df['model_idx'].isin(valid_model_idxs)]
    # Continue with strict-specific processing
    df = df.drop(columns=['atom_name']).drop_duplicates()
    # Sort the DataFrame for consistent ordering
    sorted_df = df.sort_values(['model_idx', 'chain_id', 'residue_number']).reset_index(drop=True)
    complete_seqs = []
    model_indices = sorted_df["model_idx"].unique()

    for model_idx in model_indices:
        # Extract unique chains for the current model_idx
        chains = sorted_df[sorted_df["model_idx"] == model_idx]["chain_id"].unique()

        if len(chains) == 1:
            # Single chain: construct the sequence
            residues = sorted_df[sorted_df["model_idx"] == model_idx]["residue_name"]
            sequence = "".join([amino3to1dict.get(res.strip(), 'X') for res in residues])
        else:
            # Multiple chains: assign 'X' as the sequence
            sequence = "X"
        complete_seqs.append(sequence)
        if seq_to_path is not None and target_dirname and ligand_id:
            # Map the sequence to its corresponding file path
            if sequence not in seq_to_path:
                seq_to_path[sequence] = [os.path.join(target_dirname, f"{ligand_id}_full_ab_{model_idx}.pdb")]
            else:
                seq_to_path[sequence].append(os.path.join(target_dirname, f"{ligand_id}_full_ab_{model_idx}.pdb"))

    return complete_seqs


def extract_natoms_from_pdb(pdb_file):
    sdf_file = pdb_file.replace(".pdb", ".sdf")
    if not os.path.exists(sdf_file):
        return []
    mols = Chem.SDMolSupplier(sdf_file)
    res = []
    for mol in mols:
        if mol is None:
            res.append(0)
        res.append(mol.GetNumAtoms())
    return res
