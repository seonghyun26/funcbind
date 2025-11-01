import numpy as np
import os
from rdkit import Chem
from funcbind.metrics.metrics import MetricsSampling
from funcbind.metrics.metrics_crossdocked import dock
import random
from datetime import datetime
import os
from collections import deque, defaultdict
import re
import warnings
import math
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from openbabel import openbabel
from openbabel import pybel
from funcbind.metrics.NCAA_lib import residue_atoms, N_methyl

import wandb
import pandas as pd
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

class MetricsSamplingMCPP(MetricsSampling):
    def __init__(self, config, target_dirname=None, df_mol=None, mol=None, rdkmol=None, valid_mols=None):
        super().__init__(config, target_dirname)
        # Support both legacy df_mol usage and new mol-based usage
        self.df_mol = df_mol
        self.mol = mol
        self.rdkmol = rdkmol
        self.valid_mols = valid_mols if valid_mols is not None else 0
        self.target_dir = target_dirname
        self.general_stats = []
        self.res_mcpp_metrics = []
        self._parameters = config.get("metrics_params", {})
        self.general = True
        self.mcpp_metrics = True
        self._temp_cleanup_dir = None

    def update(self, ligand_gt: dict = None, receptor: dict = None, verbose=True, **kwargs):
        # Determine which mode to use based on available data
        if self.mol is not None and self.rdkmol is not None:
            # Use mol-based converter (from metrics_mcpp_sampling.py)
            converter = SDFConverter(mol=self.mol, rdkmol=self.rdkmol, valid_mols=self.valid_mols)
            timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            random_suffix = "-" + str(random.randint(0, 999999999999)).zfill(4)
            timestamp = timestamp_prefix + random_suffix
            df_mol = converter.mol_to_sequence(
                output_dir=self.target_dir, filter_unknown_aas=True, filter_non_cyclized=True, verbose=verbose)
            try:
                os.system(f"rm -rf ./temp_{timestamp}")
            except Exception as e:
                print(f"Error removing temp folder: {e} pass")

            # Populate molecular_stats for new mode to ensure parent class compute() works
            try:
                # Use RDKit to get accurate fragment count
                from rdkit import Chem
                n_fragments = len(Chem.rdmolops.GetMolFrags(self.rdkmol, asMols=True, sanitizeFrags=False))
            except:
                n_fragments = 1  # Default to 1 if RDKit fails

            molecular_stats = {
                "n_atoms": self.mol.OBMol.NumAtoms() if self.mol else 0,
                "n_fragments": n_fragments
            }
            self.molecular_stats.append([molecular_stats])

            # Populate res_mcpp_metrics for new mode (mol-based)
            df = df_mol  # Use the df_mol as the dataframe for metrics
        else:
            # Use legacy SDF-based converter (from metrics_mcpp.py)
            super().update()  # Call parent class update for legacy mode
            converter = SDFConverter(self.target_dirname)
            converter.collect_sdf_paths()
            seed_path = os.path.join(self.target_dirname, f"{ligand_gt['id'][0]}".replace('/', '__'))  if ligand_gt is not None else None
            pocket_path = os.path.join(self.target_dirname, f"{receptor['id'][0]}".replace('/', '__'))

            # Use the current working directory where files actually exist, not temp_dir
            # This ensures compute functions look in the right place for files created by NEW MODE
            working_dir = os.getcwd()
            df, df_countAA = converter.sdf_to_sequence(
                output_dir=working_dir, convert_to_pdb=True, compute_metrics=True, seed_path=seed_path, pocket_path=pocket_path, df_mol=self.df_mol)

            # Safe extraction with None filtering
            def safe_extract_list(column_name):
                if column_name in df.columns:
                    values = df[column_name].tolist()
                    # Filter out None values and flatten if needed
                    return [v for v in values if v is not None]
                return []

            self.res_mcpp_metrics.append({
                'lRMSD': safe_extract_list('lRMSD'),
                'iRMSD': safe_extract_list('iRMSD'),
                'TM_score': safe_extract_list('TM_score'),
                'TS_full': safe_extract_list('TS_full'),
                'Ratio_>0.5_TSPR': safe_extract_list('Ratio_>0.5_TSPR'),
                'sequence': safe_extract_list('sequence'),
            })

            # Safe extraction for df_countAA
            def safe_extract_countAA(column_name):
                if column_name in df_countAA.columns:
                    values = df_countAA[column_name].tolist()
                    # Filter out None values and flatten if needed
                    return [v for v in values if v is not None]
                return []

            self.general_stats.append({
                "cyclization": safe_extract_countAA("cyclization"),
                'L_CAA': safe_extract_countAA('L_CAA'),
                'D_CAA': safe_extract_countAA('D_CAA'),
                'N_methyl': safe_extract_countAA('N_methyl'),
                'known_NCAA': safe_extract_countAA('known_NCAA'),
                'unknown_NCAA': safe_extract_countAA('unknown_NCAA'),
                'unreasonable_AA': safe_extract_countAA('unreasonable_AA')
            })

        return df

    def save(self, name="metrics.pt"):
        torch.save(
            {"molecular_stats": self.molecular_stats, "res_mcpp_metrics": self.res_mcpp_metrics, "general_stats": self.general_stats},
            os.path.join(self.target_dirname, name)
        )

    def reset(self):
        super().reset()
        self.general_stats = []
        self.res_mcpp_metrics = []

    def compute(self, **kwargs):
        # Return None only if ALL metrics collections are empty
        # In new mode, molecular_stats might be empty but we should still return results
        if (len(self.molecular_stats) == 0 and
            len(self.general_stats) == 0 and
            len(self.res_mcpp_metrics) == 0):
            return None
        res = super().compute() or {}
        res.update({
            'sequence': [x for r in self.res_mcpp_metrics for x in r['sequence']],
            'lRMSD': [x for r in self.res_mcpp_metrics for x in r['lRMSD']],
            'iRMSD': [x for r in self.res_mcpp_metrics for x in r['iRMSD']],
            'TM_score': [x for r in self.res_mcpp_metrics for x in r['TM_score']],
            'TS_full': [x for r in self.res_mcpp_metrics for x in r['TS_full']],
            'Ratio_>0.5_TSPR': [x for r in self.res_mcpp_metrics for x in r['Ratio_>0.5_TSPR']],
        })
        res.update({
            "cyclization": [x for r in self.general_stats for x in r['cyclization']],
            'L_CAA': [x for r in self.general_stats for x in r['L_CAA']],
            'D_CAA': [x for r in self.general_stats for x in r['D_CAA']],
            'N_methyl': [x for r in self.general_stats for x in r['N_methyl']],
            'known_NCAA': [x for r in self.general_stats for x in r['known_NCAA']],
            'unknown_NCAA': [x for r in self.general_stats for x in r['unknown_NCAA']],
            'unreasonable_AA': [x for r in self.general_stats for x in r['unreasonable_AA']]
        })
        return res

    def evaluate_per_pocket_metrics(self, receptor_id, res):
        table_PR = wandb.Table(columns=[
            "sequence", "lRMSD", "iRMSD", "TM_score", "TS_full", "Ratio_>0.5_TSPR",
        ])

        # Ensure that all fields are lists of the same length
        num_samples = len(res["sequence"])
        for i in range(num_samples):
            # Safety check: ensure metric lists exist and are not empty
            def safe_get_metric(metric_list, index, default_value="N/A"):
                if index < len(metric_list) and len(metric_list[index]) > 0:
                    return metric_list[index][0]
                return default_value

            sequence_str = " ".join(res["sequence"][i])
            lrmsd_val = safe_get_metric(res.get("lRMSD", []), i)
            irmsd_val = safe_get_metric(res.get("iRMSD", []), i)
            tm_score_val = safe_get_metric(res.get("TM_score", []), i)
            ts_full_val = safe_get_metric(res.get("TS_full", []), i)
            ratio_val = safe_get_metric(res.get("Ratio_>0.5_TSPR", []), i)

            table_PR.add_data(
                sequence_str,
                lrmsd_val,
                irmsd_val,
                tm_score_val,
                ts_full_val,
                ratio_val,
            )

        # Handle None receptor_id
        receptor_name = receptor_id if receptor_id is not None else "unknown"
        chart_name = f"metrics_target_{receptor_name}"

        try:
            wandb.log({chart_name: table_PR})
        except Exception as e:
            print(f"ERROR: Failed to log wandb table: {e}")

    def log_results(self, fabric, t0, docking_mode, receptor_id=None, n_atoms_gt=None):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res_log, res = {}, {}

        fabric.print("---------------- All designs results----------------")
        res_log, res = super().log_results(fabric, t0, docking_mode, receptor_id, n_atoms_gt)
        def safe_mean(values):
            # Flatten one level of nesting
            flattened = [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]
            filtered = [x for x in flattened if isinstance(x, (float, int))]
            return np.mean(filtered) if filtered else float('nan')

        self.evaluate_per_pocket_metrics(receptor_id,res)

        res_log.update({
            "lRMSD": safe_mean(res.get('lRMSD', [])),
            "iRMSD": safe_mean(res.get('iRMSD', [])),
            "TM_score": safe_mean(res.get('TM_score', [])),
            "TS_full": safe_mean(res.get('TS_full', [])),
            "Ratio_>0.5_TSPR": safe_mean(res.get('Ratio_>0.5_TSPR', [])),
        })
        fabric.print(f"lRMSD: {res_log['lRMSD']:.2f}, iRMSD: {res_log['iRMSD']:.2f}, TM_score: {res_log['TM_score']:.2f}, "
                     f"TS_full: {res_log['TS_full']:.2f}, "
                     f"Ratio_>0.5_TSPR: {res_log['Ratio_>0.5_TSPR']:.2f}, "
                     )

        total_AA = sum(res['L_CAA']) + sum(res['D_CAA']) + sum(res['N_methyl']) + sum(res['known_NCAA']) + sum(res['unknown_NCAA'])

        res_log.update({
            "cyclization_frac": sum(res['cyclization'])/len(res['cyclization']),
            "L_CAA_frac": sum(res['L_CAA'])/total_AA,
            "D_CAA_frac": sum(res['D_CAA'])/total_AA,
            "N_methyl_frac": sum(res['N_methyl'])/total_AA,
            "known_NCAA_frac": sum(res['known_NCAA'])/total_AA,
            "unknown_NCAA_frac": sum(res['unknown_NCAA'])/total_AA,
            "unreasonable_AA_frac": sum(res['unreasonable_AA'])/total_AA
        })
        fabric.print(f"cyclization_frac: {res_log['cyclization_frac']:.2f}, L_CAA_frac: {res_log['L_CAA_frac']:.2f}, "
                     f"D_CAA_frac: {res_log['D_CAA_frac']:.2f}, N_methyl_frac: {res_log['N_methyl_frac']:.2f}, "
                     f"known_NCAA_frac: {res_log['known_NCAA_frac']:.2f}, unknown_NCAA_frac: {res_log['unknown_NCAA_frac']:.2f}, "
                     f"unreasonable_AA_frac: {res_log['unreasonable_AA_frac']:.2f}")

        return res_log, res


def backbone_atoms(input_pdb):
    """
    Returns backbone atom coordinates (N, CA, C, O).

    Parameters:
        input_pdb - input pdb file

    Returns:
        backbone_coords - list of (residue number, atom name, (x, y, z)) entries
    """

    mol = next(pybel.readfile("pdb", input_pdb))

    backbone_coords = []
    res_nums = []
    index = 0
    for atom in mol.atoms:
        atom_name = atom.OBAtom.GetResidue().GetAtomID(atom.OBAtom).strip()
        if atom_name in {"N", "CA", "C", "O"}:
            res_num = atom.OBAtom.GetResidue().GetNum()
            res_name = atom.OBAtom.GetResidue().GetName()
            if res_num not in res_nums:
                index += 1
                res_nums.append(res_num)
            backbone_coords.append((index, res_name, atom_name, atom.coords))  # (residue number, atom name, (x, y, z))

    return backbone_coords


def backbone_atoms_iRMSD(input_pdb):
    mol = next(pybel.readfile("pdb", input_pdb))

    backbone_coords = []
    for atom in mol.atoms:
        atom_name = atom.OBAtom.GetResidue().GetAtomID(atom.OBAtom).strip()
        if atom_name in {"N", "CA", "C", "O"}:
            res_num = atom.OBAtom.GetResidue().GetNum()
            backbone_coords.append((res_num, atom_name, atom.coords))  # (residue number, atom name, (x, y, z))

    return backbone_coords


def match_atoms_based_on_CA(seed_dict, sample_dict):
    """
    Matches atoms between two structures based on the closeness of their CA atoms.

    Parameters:
        seed_dict - dict of backbone atom coordinates for the seed structure
        sample_dict - dict of backbone atom coordinates for the sample structure

    Returns:
        seed_xyz - NumPy array of coordinates from seed structure (matched atoms)
        sample_xyz - NumPy array of coordinates from sample structure (matched atoms)
    """

    seed_ca = seed_dict['CA']
    sample_ca = sample_dict['CA']

    # Determine which has fewer residues
    if len(seed_ca) < len(sample_ca):
        low_dict, high_dict = seed_dict, sample_dict
        low_ca, high_ca = seed_ca, sample_ca
        seed_low = True
    else:
        low_dict, high_dict = sample_dict, seed_dict
        low_ca, high_ca = sample_ca, seed_ca
        seed_low = False

    matched_indices = []

    # For each CA in the smaller set, find the closest CA in the larger set
    used_high_indices = set()
    for i, low_coord in enumerate(low_ca):
        min_dist = float('inf')
        min_j = -1
        for j, high_coord in enumerate(high_ca):
            if j in used_high_indices:
                continue  # Prevent double-matching
            dist = np.linalg.norm(np.array(low_coord) - np.array(high_coord))
            if dist < min_dist:
                min_dist = dist
                min_j = j
        matched_indices.append((i, min_j))
        used_high_indices.add(min_j)

    # Extract full backbone coordinates for matched residues
    atom_names = ['N', 'CA', 'C', 'O']
    seed_backbone = []
    sample_backbone = []

    for low_i, high_j in matched_indices:
        try:
            if seed_low:
                seed_atoms = tuple(low_dict[atom][low_i] for atom in atom_names)
                sample_atoms = tuple(high_dict[atom][high_j] for atom in atom_names)
            else:
                sample_atoms = tuple(low_dict[atom][low_i] for atom in atom_names)
                seed_atoms = tuple(high_dict[atom][high_j] for atom in atom_names)

            # Only add if all atoms are present
            if len(seed_atoms) == 4 and len(sample_atoms) == 4:
                seed_backbone.append(seed_atoms)
                sample_backbone.append(sample_atoms)
        except IndexError:
            continue  # Skip if any atom is missing

    return seed_backbone, sample_backbone


def compute_tm_score(seed_backbone, sample_backbone):
    def flatten_backbone(backbone):
        return np.array([atom for residue in backbone for atom in residue], dtype=np.float64)

    coords_seed = flatten_backbone(seed_backbone)
    coords_sample = flatten_backbone(sample_backbone)

    # Center
    seed_center = coords_seed.mean(axis=0)
    sample_center = coords_sample.mean(axis=0)
    l_A = coords_seed - seed_center
    l_B = coords_sample - sample_center

    # Kabsch alignment
    def Kabsch(A,B):
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        B_aligned = B @ R

        # Distances
        distances = np.linalg.norm(A - B_aligned, axis=1)
        L = len(distances)

        # Smoothed d0 for short fragments
        d0 = 1.24 * max(L - 15, 1)**(1/3) - 1.8
        d0 = max(d0, 1.0)
        return(distances,L,d0)

    l_distances,l_L,l_d0 = Kabsch(A=l_A,B=l_B)

    tm_score = np.sum(1 / (1 + (l_distances / l_d0)**2)) / l_L
    l_rmsd = np.sqrt(np.mean(l_distances**2))

    return tm_score,l_rmsd


def build_dict(coords):
    atom_dict = {'N': [], 'CA': [], 'C': [], 'O': []}
    for resnum, res_name, atomname, xyz in coords:
        atom_dict[atomname].append(xyz)
    return atom_dict


def compute_lRMSD_TM(df, seed_path, output_dir=None):
    """
    Calculate backbone TM_score and lRMSD using Kabsch alignment,
    and iRMSD (interface RMSD) without alignment.
    """
    for i, Name in df['Name'].items():
        if Name[0][0]:
            sample_path = Name[0][0]

            if not os.path.exists(sample_path):
                print(f"[WARNING] Missing file: {sample_path}")
                continue

            coords_seed = backbone_atoms(seed_path)
            coords_sample = backbone_atoms(sample_path)

            seed_dict = build_dict(coords_seed)
            sample_dict = build_dict(coords_sample)

            seed_backbone, sample_backbone = match_atoms_based_on_CA(seed_dict, sample_dict)

            if len(seed_backbone) == 0:
                df.at[i, 'lRMSD'] = [[None]]
                df.at[i, 'TM_score'] = [[None]]
                return df

            TM_score, l_RMSD = compute_tm_score(seed_backbone, sample_backbone)

            # Save the values in the corresponding row
            df.at[i, 'lRMSD'] = [[round(l_RMSD,2)]]
            df.at[i, 'TM_score'] = [[round(TM_score,2)]]
        else:
            df.at[i, 'lRMSD'] = [[None]]
            df.at[i, 'TM_score'] = [[None]]
            continue

    return df


def combine_pdb(MCP_pdb, protien_pdb, combined_name, cutoff=10.0):
    def extract_relevant_lines(pdb_lines):
        return [line for line in pdb_lines if line.startswith(('ATOM', 'HETATM'))]

    def get_xyz(line):
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            return (x, y, z)
        except ValueError:
            return None

    def within_cutoff(line, center, cutoff):
        coords = get_xyz(line)
        if coords is None:
            return False
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coords, center)))
        return dist <= cutoff

    # Read and filter MCP PDB
    with open(MCP_pdb, 'r') as f1:
        lines1 = f1.readlines()
    atoms1 = extract_relevant_lines(lines1)

    # Compute geometric center of MCP
    MCP_coords_all = [get_xyz(line) for line in atoms1 if get_xyz(line) is not None]
    center = np.mean([c for c in MCP_coords_all if c is not None], axis=0)

    # Add TER to separate chains
    if atoms1:
        atoms1.append("TER\n")

    # Read and filter protein PDB
    with open(protien_pdb, 'r') as f2:
        lines2 = f2.readlines()
    atoms2 = extract_relevant_lines(lines2)

    # Combine
    combined = atoms1 + atoms2

    # Filter by distance if center is provided
    if center is not None:
        combined = [line for line in combined if within_cutoff(line, center, cutoff)]

    combined.append("END\n")

    # Write to output
    with open(combined_name, 'w') as fout:
        fout.writelines(combined)


def kabsch_irmsd(P, Q):
    """
    Perform Kabsch algorithm to find the best rotation matrix
    that aligns P onto Q.
    """
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)
    return U


def compute_iRMSD(df,output_dir, seed_path, pocket_path):
    """
    Calculate backbone iRMSD between a sample and a seed, matching atoms by name,
    using Kabsch algorithm for best alignment.
    """
    # Combine PDBs with cutoff
    seed_combined = os.path.join(output_dir,'seed_combined.pdb')
    sample_combined = os.path.join(output_dir,'sample_combined.pdb')
    combine_pdb(MCP_pdb=seed_path,
                protien_pdb=pocket_path,
                combined_name=seed_combined,
                cutoff=15.0)

    for i, Name in df['Name'].items():
        if Name[0][0]:
            sample_path = Name[0][0]

            if not os.path.exists(sample_path):
                print(f"[WARNING] Missing file: {sample_path}")
                continue

            combine_pdb(MCP_pdb=sample_path,
                        protien_pdb=pocket_path,
                        combined_name=sample_combined,
                        cutoff=15.0)

            coords_seed = backbone_atoms_iRMSD(seed_combined)
            coords_sample = backbone_atoms_iRMSD(sample_combined)

            # Build dictionaries: {residue_atom -> xyz}
            def build_dict(coords):
                atom_dict = {}
                for resnum, atomname, xyz in coords:
                    key = f"{resnum}_{atomname}"
                    atom_dict[key] = xyz
                return atom_dict

            seed_dict = build_dict(coords_seed)
            sample_dict = build_dict(coords_sample)

            # Match common atoms
            matched_atoms = [key for key in sample_dict.keys() if key in seed_dict]

            if len(matched_atoms) == 0:
                print("No matching backbone atoms found!")
                continue

            xyz_seed = np.array([seed_dict[key] for key in matched_atoms])
            xyz_sample = np.array([sample_dict[key] for key in matched_atoms])

            # Center the coordinates
            centroid_seed = np.mean(xyz_seed, axis=0)
            centroid_sample = np.mean(xyz_sample, axis=0)

            P = xyz_seed - centroid_seed
            Q = xyz_sample - centroid_sample

            # Find the best rotation matrix
            U = kabsch_irmsd(P, Q)
            P_rotated = np.dot(P, U)

            # Compute RMSD
            diff = P_rotated - Q
            i_RMSD = np.sqrt(np.sum(diff**2) / len(matched_atoms))
            df.at[i, 'iRMSD'] = [[round(i_RMSD,2)]]
        else:
            df.at[i, 'iRMSD'] = [[None]]
            continue

    return df


def load_molecule_from_pdb(pdb_file):
    """Load a molecule from a PDB file."""
    return next(pybel.readfile("pdb", pdb_file))


def calculate_tanimoto(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints."""
    return fp1 | fp2  # "|" operator computes Tanimoto similarity in Pybel


def compute_TS_full(df, seed_path, output_dir=None):
    mol_seed = load_molecule_from_pdb(seed_path)
    seed_fp = mol_seed.calcfp()

    for i, Name in df['Name'].items():
        if Name[0][0]:
            sample_path = Name[0][0]

            if not os.path.exists(sample_path):
                print(f"[WARNING] Missing file: {sample_path}")
                continue
            mols_sample = load_molecule_from_pdb(sample_path)

            sample_fp = mols_sample.calcfp()
            TS_full = calculate_tanimoto(sample_fp, seed_fp)

            df.at[i, 'TS_full'] = [[round(TS_full,2)]]
        else:
            df.at[i, 'TS_full'] = [[None]]
            continue

    return(df)


def compute_AA_metrics(df):
    df_countAA = pd.DataFrame(columns = ['Name','cyclization','L_CAA','D_CAA','N_methyl','known_NCAA', 'unknown_NCAA','unreasonable_AA'])
    for i, row in df[['Name', 'cyclization', 'sequence', 'unreasonable']].iterrows():
        if row['Name'][0] is not None:
            sequence = row['sequence']
            unreasonable = row['unreasonable']
            AA_labels = {'L_CAA': 0,'D_CAA': 0,'N_methyl': 0,'known_NCAA': 0,'unknown_NCAA': 0,'unreasonable_AA': 0}
            for AA in sequence:
                found = False
                for L_AA,atoms in residue_atoms.items():
                    if AA == L_AA:
                        AA_labels['L_CAA'] += 1
                        found = True
                        break
                if not found:
                    if AA[0] == 'D':
                        AA_labels['D_CAA'] += 1
                    elif AA[0] == 'M':
                        AA_labels['N_methyl'] += 1
                    elif AA[0] == 'U':
                        AA_labels['unknown_NCAA'] += 1
                    else:
                        AA_labels['known_NCAA'] += 1
            if unreasonable != {}:
                AA_labels['unreasonable_AA'] += len(unreasonable)

            # New row as dict
            new_row = {"Name": row['Name'][0], "cyclization": row['cyclization'][0], 'L_CAA': AA_labels['L_CAA'], 'D_CAA': AA_labels['D_CAA'], 'N_methyl': AA_labels['N_methyl'],'known_NCAA': AA_labels['known_NCAA'],'unknown_NCAA': AA_labels['unknown_NCAA'],'unreasonable_AA': AA_labels['unreasonable_AA']}

            df_countAA = pd.concat([df_countAA, pd.DataFrame([new_row])], ignore_index=True)

    return df_countAA


def calculate_percent_AAs(df_countAAs):
    # Initialize output DataFrame
    df_percentAAs = pd.DataFrame(columns=[
        'cyclization', 'L_CAA', 'D_CAA', 'N_methyl',
        'known_NCAA', 'unknown_NCAA', 'unreasonable_AA'
    ])

    # Sum counts
    totals = {
        'L_CAA': df_countAAs['L_CAA'].sum(),
        'D_CAA': df_countAAs['D_CAA'].sum(),
        'N_methyl': df_countAAs['N_methyl'].sum(),
        'known_NCAA': df_countAAs['known_NCAA'].sum(),
        'unknown_NCAA': df_countAAs['unknown_NCAA'].sum(),
        'unreasonable_AA': df_countAAs['unreasonable_AA'].sum()
    }

    total_AAs = sum([totals[k] for k in ['L_CAA', 'D_CAA', 'N_methyl', 'known_NCAA', 'unknown_NCAA']])

    # Prevent division by zero
    if total_AAs == 0:
        percentages = {k: 0.0 for k in totals}
    else:
        percentages = {
            k: round((v / total_AAs) * 100, 1)
            for k, v in totals.items()
        }

    # Cyclization mean as a percent
    cyc_percent = round(df_countAAs['cyclization'].mean() * 100, 1)

    # Compose final row
    new_row = {'cyclization': cyc_percent, **percentages}
    df_percentAAs.loc[0] = new_row

    return df_percentAAs


def split_pdb_by_residue(pdb_file):
    """
    Splits a PDB file into individual residues and returns:
    - A list of Pybel Molecule objects (one per residue)
    - A list of residue names in sequence (from lines with ' N ')
    """
    with open(pdb_file, "r") as file:
        lines = file.readlines()

    residues = {}
    current_res_id = None
    current_residue_lines = []
    sequence = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            res_id = line[17:27]  # Includes residue name + chain + number
            res_name = line[17:20]

            if res_id != current_res_id:
                if current_res_id is not None:
                    residues[current_res_id] = current_residue_lines
                current_res_id = res_id
                current_residue_lines = []
                sequence.append(res_name)

            current_residue_lines.append(line)

    # Add the last residue
    if current_res_id and current_residue_lines:
        residues[current_res_id] = current_residue_lines

    mol_list = []

    for i, (res_id, res_lines) in enumerate(residues.items()):
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = "-" + str(random.randint(0, 999999999999)).zfill(4)
        timestamp = timestamp_prefix + random_suffix
        residue_pdb = f"temp_residue_{i}_{timestamp}.pdb"
        with open(residue_pdb, "w") as res_file:
            res_file.writelines(res_lines)

        mol = next(pybel.readfile("pdb", residue_pdb), None)
        if mol:
            mol_list.append(mol)

        # Delete the temporary PDB file
        os.remove(residue_pdb)

    return mol_list, sequence


def compute_similarity(seed_mols, seed_seq, sample_mols, sample_seq):
    similarity = []
    AA_similar = 0
    for i in range(len(sample_mols)):
        sdf_fp = sample_mols[i].calcfp()
        pdb_fp = seed_mols[i].calcfp()
        sim = calculate_tanimoto(sdf_fp, pdb_fp)
        similarity.append(sim)
        if (sample_seq[i] == seed_seq[i] or
            (sample_seq[i].startswith('D') and sample_seq[i][1:] == seed_seq[i][1:])):
            AA_similar += 1
    return AA_similar, similarity, sample_seq


def compute_per_res_ratio(df, seed_path, output_dir=None):
    for i, Name in df['Name'].items():
        if Name[0][0]:
            sample_path = Name[0][0]

            if not os.path.exists(sample_path):
                print(f"[WARNING] Missing file: {sample_path}")
                continue

            mol_seed_res, seed_seq = split_pdb_by_residue(seed_path)
            mol_sample_res, sample_seq = split_pdb_by_residue(sample_path)

            crystal_size = len(mol_seed_res)
            sample_size = len(mol_sample_res)

            similarity_high = []

            if sample_size >= crystal_size:
                # Normal forward windows
                for start in range(sample_size - crystal_size + 1):
                    sample_window = mol_sample_res[start:start+crystal_size]
                    seq_window = sample_seq[start:start+crystal_size]
                    AA_similar, similarity, aligned_seq = compute_similarity(mol_seed_res, seed_seq, sample_window, seq_window)
                    if sum(similarity) > sum(similarity_high):
                        similarity_high = similarity

                # Reverse windows
                for end in range(crystal_size, sample_size + 1):
                    sample_window = mol_sample_res[end-crystal_size:end][::-1]
                    seq_window = sample_seq[end-crystal_size:end][::-1]
                    AA_similar, similarity, aligned_seq = compute_similarity(mol_seed_res, seed_seq, sample_window, seq_window)
                    if sum(similarity) > sum(similarity_high):
                        similarity_high = similarity

            else:
                # Partial forward alignment
                crystal_subsets = [mol_seed_res[i:i+sample_size] for i in range(crystal_size - sample_size + 1)]
                crystal_seq_subsets = [seed_seq[i:i+sample_size] for i in range(crystal_size - sample_size + 1)]

                for crystal_sub, crystal_seq_sub in zip(crystal_subsets, crystal_seq_subsets):
                    AA_similar, similarity, aligned_seq = compute_similarity(crystal_sub, crystal_seq_sub, mol_sample_res, sample_seq)
                    if sum(similarity) > sum(similarity_high):
                        similarity_high = similarity

                # Partial reverse alignment
                reversed_sample_mols = mol_sample_res[::-1]
                reversed_sequence = sample_seq[::-1]

                for crystal_sub, crystal_seq_sub in zip(crystal_subsets, crystal_seq_subsets):
                    AA_similar, similarity, aligned_seq = compute_similarity(crystal_sub, crystal_seq_sub, reversed_sample_mols, reversed_sequence)
                    if sum(similarity) > sum(similarity_high):
                        similarity_high = similarity

            rounded_TS_PR = [round(x, 3) for x in similarity_high]
            TS_above = sum(x >= 0.5 for x in rounded_TS_PR)
            ratio_TS_above = round(TS_above / len(rounded_TS_PR), 2) if len(rounded_TS_PR) > 0 else 0

            df.at[i, 'TS_per_residue'] = [rounded_TS_PR]
            df.at[i, 'Ratio_>0.5_TSPR'] = [[ratio_TS_above]]
        else:
            df.at[i, 'TS_per_residue'] = [[None]]
            df.at[i, 'Ratio_>0.5_TSPR'] = [[None]]
            continue

    return df


class SDFConverter:
    def __init__(self, data_dir=None, mol=None, rdkmol=None, valid_mols=None):
        # Support both legacy data_dir mode and new mol-based mode
        self.data_dir = data_dir
        self.samples_paths = []
        self.mol = mol
        self.rdkmol = rdkmol
        self.valid_mols = valid_mols

    def print_last_row_values(self,df):
        last_row = df.iloc[-1]
        for column in df.columns:
            print(f"{column}: {last_row[column]}")

    def unwrap_outer_list(self,val):
        if isinstance(val, list) and len(val) == 1:
            if isinstance(val[0], list) or isinstance(val[0], dict) or isinstance(val[0], (str, int, float)):
                return val[0]
        return val

    def Update_NCAA_dict(self,new_entries):
        with open("NCAA_lib.py", "a") as f:
            for key, value in new_entries.items():
                f.write(f'\nNCAA_dict["{key}"] = "{value}"')

        # Reload the updated dictionary
        from importlib import reload
        from funcbind.metrics import NCAA_lib
        reload(NCAA_lib)

    def Find_NCAA_num(self):
        with open("NCAA_lib.py", "r") as f:
            for line in f:
                if 'NCAA_dict["' in line:
                    numbers = re.findall(r'\d+', line)
                    NCAA_num = int(numbers[-1]) + 1 if numbers else 1
        return(NCAA_num)

    def collect_sdf_paths(self):
        for samples in os.listdir(self.data_dir):
            samples_path = os.path.join(self.data_dir, samples)
            if samples_path.endswith('.sdf'):
                self.samples_paths.append(samples_path)
        return self.samples_paths

    def run_dock(self, mol, pocket_path):
        target_dirname = os.path.dirname(pocket_path)
        vina_results = dock(mol = mol,
            ligand_gt_fn = pocket_path,
            protein_root = target_dirname,
            center = None,
            docking_mode = "vina_score",
            exhaustiveness = 32)

        score_only_affinity = vina_results['score_only'][0]['affinity']
        minimize_affinity = vina_results['minimize'][0]['affinity']

        return(score_only_affinity,minimize_affinity)

    def sdf_to_pdb(self, sdf_path):
        mol = next(pybel.readfile("sdf", sdf_path))
        mol.make3D()
        pdb_str = mol.write("pdb")
        return pdb_str

    def is_cyclic(self, smiles):
        """
        Determines if a given SMILES string represents a cyclic compound.

        Parameters:
            smiles (str): The SMILES string to check.

        Returns:
            bool: True if the molecule is cyclic, False otherwise.
        """
        try:
            molecule = pybel.readstring("smi", smiles)
            obmol = molecule.OBMol
            obmol.FindSSSR()  # Ensure rings are detected

            for ring in obmol.GetSSSR():
                if ring.Size() >= 10:
                    return True  # Found a ring with 10 or more atoms
            return False  # No such ring found
        except Exception as e:
            print(f"Error processing SMILES string: {e}")
            return None


    def group_pdb_by_residue(self,pdb_lines):
        residues = {}
        res_num_list = []
        connects = []
        import re

        def fix_pdb_line(line: str) -> str:

            # Insert space before any '-' that follows a digit to fix smashed coordinates
            line = re.sub(r'(?<=\d)-', ' -', line)

            parts = line.split()
            if len(parts) < 11:
                return line

            # Extract expected parts
            atom_serial = int(parts[1])
            atom_name = parts[2]
            res_name = parts[3][:3]
            chain_id = parts[4]
            res_seq = int(parts[5])
            x, y, z = map(float, parts[6:9])
            occupancy = float(parts[9])
            bfactor = float(parts[10])
            element = parts[11].strip().upper()

            # Pad/align atom name: if 4 characters, no pad; else right-align

            # Format line using proper PDB fixed-width columns
            pdb_line = (
                f"ATOM  "
                f"{atom_serial:5d}  "
                f"{atom_name:<4}"
                f"{res_name:>3} "
                f"{chain_id:1}"
                f"{res_seq:4d}"
                f"    "
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"{occupancy:6.2f}"
                f"{bfactor:6.2f}"
                f"          "
                f"{element:>2}"
                f"\n"
            )

            return pdb_line

        for line in pdb_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = fix_pdb_line(line)
                try:
                    residue_id = round(float(line[22:26].strip()))
                except:
                    residue_id = round(float(line[23:27].strip()))
                residues.update({line: residue_id})
                if residue_id not in res_num_list:
                    res_num_list.append(residue_id)
            if line.startswith("CONECT"):
                connects.append(line)
        return residues, res_num_list, connects


    def reorder_combine_pdb(self,df_mol,state_pdb_files, verbose=True):

        count_SM = 0
        for s in range(0,len(state_pdb_files)):
            state_pdb_file = state_pdb_files[s]
            if state_pdb_file == 'small_molecule':
                count_SM += 1
                continue
            clean_pdb_name = state_pdb_file.replace(".pdb", f"_clean.pdb")
            df_mol.at[s, 'Name'] = [[clean_pdb_name]]

            with open(clean_pdb_name,'w') as clean_file: # Add pdb lines in order to combined file
                clean_file.truncate(0)

            with open(state_pdb_file, "r") as file:
                pdb_lines = file.readlines()
            residues,res_num_list,connects = self.group_pdb_by_residue(pdb_lines)

            sorted_res_num_list = sorted(res_num_list)

            with open(clean_pdb_name,'a') as clean_file: # Add pdb lines in order to combined file
                for res in sorted_res_num_list:
                    for line, residue_id in residues.items():
                        if residue_id == res:
                            clean_file.write(line)
                for connect in connects:
                    clean_file.write(connect)
            os.remove(state_pdb_file)
        if verbose:
            print('number of small molecules: ',count_SM)

        return(df_mol)


    def seq_to_pdb(self, df_mol, output_dir):
        """
        Convert an SDF file with multiple states into separate PDB files,
        assigning residue names based on provided sequences.

        Args:
            sequences (list): List of amino acid sequences (one per state).
            sdf_file (str): Path to the input SDF file.
            atom_indices_all (list of dict): Atom indices for each labeled atom in residues (N, CA, CB, C, O) for each state.
            ordered_residues_all (list of list): Ordered atom indices for each residue per state.

        Returns:
            list: Paths to the generated PDB files.
        """

        mol = self.rdkmol
        output_files = []

        sequences = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                    for item in df_mol['sequence'].tolist()[1:]]
        sequence = df_mol['sequence'].iloc[0][0]

        if mol is None:
            return  output_files # Skip if the molecule failed to load

        output_file = df_mol['Name'].iloc[0][0][0]

        ordered_residue_indices = df_mol['residue_indices'][0]
        atom_indices = df_mol['atom_indices'][0][0]
        if atom_indices == [None]:
            output_files.append('small_molecule')
            return  output_files

        res_names = []
        res_numbers = []
        atom_names = []
        atom_unk_count = 0
        sulfur = False
        unk_inds = []

        # Assign residue names and numbers
        for i, atom in enumerate(mol.GetAtoms()):
            found = False
            for res_ind, residue in enumerate(ordered_residue_indices):
                if (i + 1) in residue:
                    res_names.append(sequence[res_ind])
                    res_numbers.append(res_ind + 1)
                    found = True
                    break
            if not found:
                if atom.GetSymbol() == 'S':
                    sulfur = True
                res_names.append('UNK')
                res_numbers.append(len(ordered_residue_indices) + 1)
                unk_inds.append(i)

        if len(unk_inds) <= 4:
            res_names = ["UNA" if r == "UNK" else r for r in res_names]
        elif sulfur == True:
            res_names = ["UNS" if r == "UNK" else r for r in res_names]

        for unknown in ['UNK', 'UNS', 'UNA']:
            if unknown in res_names:
                sequence_extended = sequence + [unknown]
                df_mol.at[0, 'sequence'] = [sequence_extended]
                break

        # Assign atom names
        for i, atom in enumerate(mol.GetAtoms()):
            found = False
            for atom_name, atom_index in atom_indices.items():
                if (atom_index - 1) == i:
                    atom_names.append(atom_name.split(':')[0])
                    found = True
                    break
            if not found:
                atom_unk_count += 1
                atom_names.append(f"{atom.GetSymbol()}{atom_unk_count}")

        # Set PDB residue info
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetMonomerInfo(
                rdchem.AtomPDBResidueInfo(
                    atomName=atom_names[i],
                    residueName=res_names[i],
                    residueNumber=res_numbers[i],
                    chainId='A'
                )
            )

        # Generate 3D coordinates if missing
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

        Chem.rdmolfiles.MolToPDBFile(mol, output_file)
        output_files.append(output_file)

        return output_files


    def sdf_to_sequence(self, output_dir, convert_to_pdb, compute_metrics, seed_path, pocket_path, df_mol):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if compute_metrics is True and convert_to_pdb is not True:
            raise ValueError(f"Need convert to PDB to be True if compute metrics is True.")

        df = pd.DataFrame()
        try:
            NCAA_num = self.Find_NCAA_num()
        except:
            NCAA_num = 1

        for sdf_path in self.samples_paths:
            df = pd.concat([df, df_mol.iloc[0:]], ignore_index=True)
            df = df.drop(columns=['residue_indices'])
            df = df.drop(columns=['atom_indices'])
            df = df.drop(columns=['NCAA_num'])
            if seed_path is not None and os.path.exists(seed_path) and compute_metrics:
                df['lRMSD'] = [[None] for _ in range(len(df))]
                df['iRMSD'] = [[None] for _ in range(len(df))]
                df['TM_score'] = [[None] for _ in range(len(df))]
                df['TS_full'] = [[None] for _ in range(len(df))]
                df['TS_per_residue'] = [[None] for _ in range(len(df))]
                df['Ratio_>0.5_TSPR'] = [[None] for _ in range(len(df))]

            df_mol = df_mol.iloc[0:0]

            if seed_path is not None and os.path.exists(seed_path) and compute_metrics:
                df = compute_lRMSD_TM(df, seed_path, output_dir)
                df = compute_TS_full(df, seed_path, output_dir)
                df = compute_per_res_ratio(df, seed_path, output_dir)
                if os.path.exists(pocket_path):
                    df = compute_iRMSD(df,output_dir, seed_path, pocket_path)
            else:
                print(f"Skipping metrics computation: seed_path is None or invalid (got {seed_path}).")

            df = df.applymap(self.unwrap_outer_list)

            ##########  Combine PDBs  ##########
            df["Name"] = df["Name"].apply(lambda x: x[0] if isinstance(x, list) else x)
            allowed_names = list(df["Name"])
            combined_pdb = sdf_path.replace('.sdf','.pdb')
            with open(combined_pdb, "w") as f:
                pass
            for i in range(0,len(allowed_names)):
                allowed_name_path = allowed_names[i]
                with open(combined_pdb, 'a') as combined_f:
                    combined_f.write(f"MODEL {i+1}\n")
                    with open(allowed_name_path, 'r') as allowed_f:
                        for line in allowed_f:
                            if "ATOM" in line or 'HETATM' in line:
                                combined_f.write(line)
                    combined_f.write("ENDMDL\n")
            ######################################
        df_countAA = compute_AA_metrics(df)

        return df,df_countAA

    def check_residue(self,df_res):
        all_res_atom_names = []
        res_num = len(df_res)
        atom_indices = df_res['atom_indices'][res_num-1][0]
        indices_residue = df_res['indices_residue'][res_num-1][0]
        check_sequence = df_res['check_sequence'][res_num-1][0]

        for atom_name_num, index in atom_indices.items():
            atom_name = atom_name_num.split(':')[0]
            atom_res_num = atom_name_num.split(':')[-1]
            if int(atom_res_num) == int(res_num):
                all_res_atom_names.append(atom_name)
        all_res_atom_names_set = set(all_res_atom_names)

        matching_residue = [
            res for res, atoms in residue_atoms.items()
            if set(atoms) == all_res_atom_names_set
        ]

        if len(matching_residue) == 1:
            if not isinstance(indices_residue, (list, tuple)):
                indices_residue = [indices_residue]
            check_sequence.update({tuple(indices_residue): matching_residue[0]})
        df_res.at[res_num - 1, 'check_sequence'] = [check_sequence]

        return df_res

    def NCAA_check(self,smiles,NCAA_num,check_sequence,residue_index):
        from funcbind.metrics.NCAA_lib import NCAA_dict, AA3_dict

        def remove_bond_orders(smiles):
            # Remove = [ ] , @ H \ / # and spaces
            cleaned = re.sub(r"[=\[\],@H\\/#! ]", "", smiles)
            return cleaned.lower()

        Found = False
        # Check if SMILES exists in NCAA_dict
        for AA_smiles, NCAA in NCAA_dict.items():
            if AA_smiles == smiles:
                AA_name = AA3_dict.get(NCAA, NCAA)  # Use three-letter code if available
                Found = True
                break
            try:
                AA_smiles_nobondorder = remove_bond_orders(AA_smiles)
                smiles_nobondorder = remove_bond_orders(smiles)
                if AA_smiles_nobondorder == smiles_nobondorder:
                    AA_name = AA3_dict.get(NCAA, NCAA)
                    Found = True
                    break
            except:
                continue

        # Assign a new name if SMILES is unknown
        if not Found:
            for indices, name in check_sequence.items():
                if set(residue_index) == set(indices):
                    AA_name = name
                    Found = True
                    break
        if not Found:
            if 'S' in smiles or 's' in smiles:
                AA_name = 'UNS'
            else:
                AA_name = f"UK{NCAA_num}"
            NCAA_num += 1
            new_entries = {smiles: AA_name}
            self.Update_NCAA_dict(new_entries)

        return(AA_name,NCAA_num)

    def find_branching_atoms(self, df_res, mol):
        """
        Finds all atoms branching off from the CB_index atom in a molecule.

        Parameters:
            mol - Pybel molecule object
            CB_index - Index of the alpha-carbon (CB) atom
            indices_residue - List of atom indices connected to residue

        Returns:
            indices_residue - Updated list of atom indices connected to residue
        """
        res_num = len(df_res)
        def loop_branching_atom(df_res, atoms, sulfur_index, letter_count, atom_Names):
            atom_loop = []
            atom_letters = ['G', 'D', 'E', 'Z', 'H', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '1', '2', '3', '4', '5', '6', '7']
            symbol_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
            atom_len = len(atoms)
            res_num = len(df_res)
            atom_indices = df_res['atom_indices'][res_num-1][0]
            indices_residue = df_res['indices_residue'][res_num-1][0]

            total_n_indices_count = 0
            total_neighbor_atom_symbols = []
            if letter_count == 2:
                for atom in atoms:
                    for bond in openbabel.OBAtomBondIter(atom):
                        neighbor_atom = bond.GetNbrAtom(atom)
                        if neighbor_atom.GetIdx() in indices_residue:
                            total_n_indices_count += 1
                        else:
                            total_neighbor_atom_symbols.append(neighbor_atom.GetAtomicNum())

            for atom in atoms:
                bond_count = sum(1 for _ in openbabel.OBAtomBondIter(atom))
                bc = 0

                n_indices_count = 0
                neighbor_atom_symbols = []
                for bond in openbabel.OBAtomBondIter(atom):
                    neighbor_atom = bond.GetNbrAtom(atom)  # Get bonded neighbor atom
                    if neighbor_atom.GetIdx() in indices_residue:
                        n_indices_count += 1
                    else:
                        neighbor_atom_symbols.append(neighbor_atom.GetAtomicNum())

                if atom_Names == {}:
                    if len(set(neighbor_atom_symbols)) == 2 or len(set(total_neighbor_atom_symbols)) == 2:
                        if letter_count == 0:
                            if 8 in neighbor_atom_symbols and 6 in neighbor_atom_symbols:
                                atom_Names = {(8,0): 'OG1', (6,0): 'CG2'}
                        if letter_count == 1:
                            if 8 in neighbor_atom_symbols and 7 in neighbor_atom_symbols:
                                atom_Names = {(8,1): 'OD1', (7,1): 'ND2'}
                            if 6 in neighbor_atom_symbols and 7 in neighbor_atom_symbols:
                                atom_Names = {(7,1): 'ND1', (6,1): 'CD2',(7,2): 'NE2', (6,2): 'CE1'}
                        if letter_count == 2:
                            if 8 in neighbor_atom_symbols and 7 in neighbor_atom_symbols:
                                atom_Names = {(8,2): 'OE1', (7,2): 'NE2'}
                            if 6 in total_neighbor_atom_symbols and 7 in total_neighbor_atom_symbols and len(total_neighbor_atom_symbols) >= 3:
                                atom_Names = {(7,2): 'NE1', (6,2): 'CE', (6,3): 'CZ', (6,4): 'CH2', (100,100): 2}

                for bond in openbabel.OBAtomBondIter(atom):
                    neighbor_atom = bond.GetNbrAtom(atom)  # Get bonded neighbor atom
                    n_bond_count = sum(1 for _ in openbabel.OBAtomBondIter(neighbor_atom))

                    if neighbor_atom.GetIdx() not in indices_residue and neighbor_atom.GetAtomicNum() != 1:
                        indices_residue.append(neighbor_atom.GetIdx())
                        atom_symbol = neighbor_atom.GetAtomicNum()  # Get atomic number
                        for a_symbol, a_num in symbol_dict.items():
                            if a_num == atom_symbol:
                                symbol = a_symbol
                                break
                        atom_index = neighbor_atom.GetIdx() # Get atom index
                        if atom_symbol == 16:
                            sulfur_index.append(atom_index)
                        atom_loop.append(neighbor_atom)

                        if atom_Names != {}:
                            for atom_num,atom_name in atom_Names.items():
                                TRP_count = list(atom_Names.values())[-1]
                                if atom_num[0] == atom_symbol and atom_num[1] == letter_count:
                                    if len(atom_Names) >= 5 and len(atom_name) == 2:
                                        atomName = f"{atom_name}{TRP_count}:{res_num}"
                                        atom_indices.update({atomName: atom_index})
                                        atom_Names[list(atom_Names.keys())[-1]] = 3 if TRP_count == 2 else 2

                                    else:
                                        atomName = f"{atom_name}:{res_num}"
                                        atom_indices.update({atomName: atom_index})
                                    break
                        elif bond_count >= 3:
                            if n_indices_count == 1:
                                bc += 1
                                atomName = f"{symbol}{atom_letters[letter_count]}{bc}:{res_num}"
                                atom_indices.update({atomName: atom_index})
                            else:
                                atomName = f"{symbol}{atom_letters[letter_count]}:{res_num}"
                                atom_indices.update({atomName: atom_index})
                        else:
                            if atom_len >= 2:
                                neighbor_indices_count = 0

                                for bond2 in openbabel.OBAtomBondIter(neighbor_atom):
                                    neighbor_atom2 = bond2.GetNbrAtom(neighbor_atom)  # Get bonded neighbor atom
                                    if neighbor_atom2.GetIdx() in indices_residue:
                                        neighbor_indices_count += 1
                                if neighbor_indices_count >= 2:
                                    atomName = f"{symbol}{atom_letters[letter_count]}:{res_num}"
                                    atom_indices.update({atomName: atom_index})
                                else:
                                    branch_num = None

                                    for name, index in atom_indices.items():
                                        if index == atom.GetIdx():
                                            branch_name = name.split(':')[0]
                                            branch_num = branch_name[-1]
                                            break
                                    if branch_num is not None and branch_num.isnumeric():
                                        atomName = f"{symbol}{atom_letters[letter_count]}{branch_num}:{res_num}"
                                        atom_indices.update({atomName: atom_index})
                                    else:
                                        atomName = f"{symbol}{atom_letters[letter_count]}:{res_num}"
                                        atom_indices.update({atomName: atom_index})
                            else:
                                atomName = f"{symbol}{atom_letters[letter_count]}:{res_num}"
                                atom_indices.update({atomName: atom_index})

                        # Get bond length
                        bond_length = bond.GetLength()
                        if bond_length <= 0.8:
                            unreasonable_bond_lengths[(atom.GetIdx(), atom_index)] = round(bond_length, 3)
                        if atom_symbol == 8 and atom.GetAtomicNum() == 8:
                            unreasonable_bond_lengths[(atom.GetIdx(), atom_index)] = 'O-O bond'
                        elif atom_symbol == 7 and atom.GetAtomicNum() == 7:
                            unreasonable_bond_lengths[(atom.GetIdx(), atom_index)] = 'N-N bond'
            letter_count += 1
            df_res.at[res_num - 1, 'atom_indices'] = [atom_indices]
            df_res.at[res_num - 1, 'indices_residue'] = [indices_residue]

            return df_res, atom_loop, sulfur_index, letter_count, atom_Names

        cb_index_list = df_res['CB_index'][res_num-1][0]
        cb_atom = mol.OBMol.GetAtom(int(cb_index_list[0]))
        atom_loop = [cb_atom]
        sulfur_index = []
        unreasonable_bond_lengths = {}
        letter_count = 0
        atom_Names = {}

        while atom_loop:
            indices_residue = df_res.at[res_num-1, 'indices_residue'][0]

            if len(indices_residue) >= 20:
                if sulfur_index:
                    index = indices_residue.index(sulfur_index[0])
                    df_res.at[res_num-1, 'indices_residue'] = indices_residue[:index+1]
                break
            df_res,atom_loop,sulfur_index,letter_count,atom_Names = loop_branching_atom(
                df_res = df_res, atoms = atom_loop,  sulfur_index = sulfur_index, letter_count = letter_count, atom_Names = atom_Names
            )
        df_res = self.check_residue(df_res)
        return df_res

    def gly_residues(self, df_res, mol):
        """
        Determine any glycine residues

        Parameters:
            mol - Pybel molecule object
            df_res = dataframe

        Returns:
            df_res = updated dataframe
        """
        gly_indices = []
        len_res_indices = 0
        residue_indices = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                   for item in df_res['indices_residue'].tolist()]
        all_indices = df_res['all_indices'][len(df_res)-1][0]
        atom_indices = df_res['atom_indices'][len(df_res)-1][0]
        check_sequence = df_res['check_sequence'][len(df_res)-1][0]
        unreasonable_bond_length = df_res['unreasonable_bond_length'][len(df_res)-1][0]

        for res_index in residue_indices:
            for index in res_index:
                len_res_indices += 1
        for all_index in all_indices:
            Next_index = False  # Reset flag for each all_index
            for res_index in residue_indices:
                for index in res_index:
                    if all_index == index:
                        Next_index = True  # Correctly assign True
                        break  # Break inner loop
                if Next_index:  # If True, break outer loop
                    break
            if not Next_index:  # Append only if not found in residue_indices
                gly_indices.append(all_index)

        for atom in mol.atoms:
            res_num = len(df_res)
            atom_CAname = atom_Nname = atom_Cname = atom_Oname = atom_CBname = None

            if atom.idx in gly_indices:

                if atom.atomicnum == 6:  # Carbon only
                    atom_CAname = f"CA:{res_num}"; CA_index = atom.idx

                    for bond in openbabel.OBAtomBondIter(atom.OBAtom):  # Correct way to iterate bonds
                        neighbor_atom = bond.GetNbrAtom(atom.OBAtom)  # Get bonded neighbor
                        if neighbor_atom.GetAtomicNum() == 7:
                            atom_Nname = f"N:{res_num}"; N_index = neighbor_atom.GetIdx()
                        elif neighbor_atom.GetAtomicNum() == 6:
                            O_found = False
                            for bond2 in openbabel.OBAtomBondIter(neighbor_atom):  # Correct way to iterate bonds
                                neighbor_atom2 = bond2.GetNbrAtom(neighbor_atom)  # Get bonded neighbor
                                if O_found:
                                    break
                                if neighbor_atom2.GetAtomicNum() == 8:
                                    O_found = True
                                    atom_Cname = f"C:{res_num}"; C_index = neighbor_atom.GetIdx()
                                    atom_Oname = f"O:{res_num}"; O_index = neighbor_atom2.GetIdx()
                            if not O_found:
                                atom_CBname = f"CB:{res_num}"; CB_index = neighbor_atom.GetIdx()

                        if all(name is not None for name in [atom_CAname, atom_Nname, atom_Cname, atom_Oname]):
                            indices_residue = []
                            atom_indices.update({atom_CAname: CA_index})
                            atom_indices.update({atom_Nname: N_index})
                            atom_indices.update({atom_Cname: C_index})
                            atom_indices.update({atom_Oname: O_index})
                            indices_residue.append(CA_index); indices_residue.append(N_index);
                            indices_residue.append(C_index); indices_residue.append(O_index)
                            if atom_CBname is not None:
                                atom_indices.update({atom_CBname: CB_index})
                                indices_residue.append(CB_index)
                                new_row = {"atom_indices": [atom_indices], "indices_residue":[indices_residue], 'CB_index':[[CB_index]],
                                           'unreasonable_bond_length': [unreasonable_bond_length],'N_index': [[N_index]],'all_indices': [all_indices],
                                           'check_sequence': [check_sequence]}
                                df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
                                df_res = self.find_branching_atoms(df_res,mol)
                            else:
                                CB_index = None
                                df_res = self.check_residue(df_res)
                                new_row = {"atom_indices": [atom_indices], "indices_residue":[indices_residue], 'CB_index':[[CB_index]],
                                           'unreasonable_bond_length': [unreasonable_bond_length],'N_index': [[N_index]],'all_indices': [all_indices],
                                           'check_sequence': [check_sequence]}
                                df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
                                break

        return df_res

    def check_N_methyl(self, df_res, mol):
        residue_indices = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                   for item in df_res['indices_residue'].tolist()]
        N_index_all = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                   for item in df_res['N_index'].tolist()]
        res_index_labeled_all = {res for group in residue_indices for res in group}  # faster lookup
        check_sequence = df_res['check_sequence'][len(df_res)-1][0]

        for atom in mol.atoms:
            if atom.idx in N_index_all:
                for bond in openbabel.OBAtomBondIter(atom.OBAtom):
                    neighbor_atom = bond.GetNbrAtom(atom.OBAtom)
                    neighbor_idx = neighbor_atom.GetIdx()

                    if neighbor_idx not in res_index_labeled_all:
                        for key_tuple in list(check_sequence.keys()):
                            if atom.idx in key_tuple:
                                for group in residue_indices:
                                    if atom.idx in group:
                                        group.append(neighbor_idx)
                                        break

                                residue_id = check_sequence[key_tuple]
                                new_key = tuple(sorted(set(key_tuple + (neighbor_idx,))))
                                check_sequence[new_key] = N_methyl.get(residue_id, 'MUK')

                                # Remove the old key
                                del check_sequence[key_tuple]
                                break

        df_res.at[len(df_res) - 1, 'check_sequence'] = [check_sequence]

        return df_res

    def order_residues(self,df_res,mol,smiles):
        """
        Order residues in molfile from nitrogen atom indices and atom indexes in each residue

        Parameters:
            mol - Pybel molecule object
            residue_indices - Embedded List of atom indices connected to residue

        Returns:
            ordered_residue_indices - Updated ordered embedded list of atom indices connected to residue
        """

        def bonds_to_atom(mol, atom_index):
            atom = mol.OBMol.GetAtom(atom_index)
            neighbors = []
            for bond in openbabel.OBAtomBondIter(atom):  # Iterate over bonds
                neighbor_atom = bond.GetNbrAtom(atom)  # Get neighboring atom
                neighbors.append(neighbor_atom.GetIdx())  # Append 1-based index

            return neighbors

        def order_edges(edges, start=0):
            graph = defaultdict(list)
            for a, b in edges:
                graph[a].append(b)
                graph[b].append(a)

            all_nodes = set(graph.keys())
            ordered_nodes = []
            visited = set()
            queue = deque([start])

            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    ordered_nodes.append(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)

            # If not all nodes are visited, process remaining disconnected components
            for node in all_nodes:
                if node not in visited:
                    queue.append(node)
                    while queue:
                        node = queue.popleft()
                        if node not in visited:
                            visited.add(node)
                            ordered_nodes.append(node)
                            for neighbor in graph[node]:
                                if neighbor not in visited:
                                    queue.append(neighbor)

            return ordered_nodes

        cyclization = self.is_cyclic(smiles)
        # Get all rings using Open Babel’s SSSR (Smallest Set of Smallest Rings)
        ring_len_max = 20
        ring_systems = mol.OBMol.GetSSSR()
        MCP_ring_atoms = None
        MCP_ring_atoms_recount = []
        if df_res['N_index'][0] == [[]]:
            df_res = df_res.iloc[1:].reset_index(drop=True)
        residue_indices = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                   for item in df_res['indices_residue'].tolist()]
        N_index_all = [item[0][0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) and len(item[0]) == 1
                        else item[0] if isinstance(item, list) and len(item) == 1
                        else item
                        for item in df_res['N_index'].tolist()
                        if isinstance(item, int) or (isinstance(item, list) and item)]
        ordered_residue_indices = []

        if N_index_all == []:
            return(df_res,cyclization)

        for ring in ring_systems:
            ring_atoms = [atom for atom in ring._path]  # Get atom indices in the ring
            if len(ring_atoms) >= ring_len_max:
                MCP_ring_atoms = ring_atoms

        if MCP_ring_atoms is not None and len(MCP_ring_atoms) >= 15:
            # Identify sulfur atoms in the largest ring
            for a in range(0,len(MCP_ring_atoms)-1):
                atom_idx = MCP_ring_atoms[a]
                atom = mol.OBMol.GetAtom(atom_idx)
                if atom.GetAtomicNum() == 16:  # Sulfur atomic number is 16
                    MCP_ring_atoms1 = MCP_ring_atoms[0:a]
                    MCP_ring_atoms2 = MCP_ring_atoms[a:-1]
                    MCP_ring_atoms_recount = MCP_ring_atoms2 + MCP_ring_atoms1
                    MCP_ring_atoms = MCP_ring_atoms_recount
                    break
            for a in range(0,len(MCP_ring_atoms)):
                atom = MCP_ring_atoms[a]
                for residue_index in residue_indices:
                    if atom in residue_index:
                        if residue_index not in ordered_residue_indices:
                            ordered_residue_indices.append(residue_index)
                            break
            for residue_index in residue_indices:
                if residue_index not in ordered_residue_indices:
                    ordered_residue_indices.append(residue_index)
        else:
            connect_neighbor = []
            N_start = 0
            N_neighbors_all = []
            for n in range(0,len(N_index_all)):
                N_index = N_index_all[n]
                N_neighbors = bonds_to_atom(mol, N_index)
                N_neighbors_all.append(N_neighbors)
                if len(N_neighbors) == 1:
                    N_start = n
            if N_start != 0:
                for r in range(0,len(residue_indices)):
                    res_index = residue_indices[r]
                    for i in range(0,len(res_index)):
                        index = res_index[i]
                        for N_neighbors in N_neighbors_all:
                            for N_neighbor in N_neighbors:
                                if index == N_neighbor and r != n:
                                    connect_neighbor.append([r,n])
                ordered_nodes = order_edges(connect_neighbor, start=N_start)
                ordered_residue_indices = [residue_indices[i] for i in ordered_nodes]
            else:
                N_save_order = []
                n_previous = 0
                for N in N_index_all:
                    found = False
                    N_save_order.append(n_previous)
                    N_index = N_index_all[n_previous]
                    N_neighbors = bonds_to_atom(mol, N_index)
                    for r in range(0,len(residue_indices)):
                        if found:
                            break
                        res_index = residue_indices[r]
                        for i in range(0,len(res_index)):
                            index = res_index[i]
                            for N_neighbor in N_neighbors:
                                if index == N_neighbor and r != n_previous:
                                    n_previous = r
                                    found = True
                                    break
                    if not found:
                        for N in range(0,len(N_index_all)):
                            if N not in N_save_order:
                                n_previous = N
                ordered_residue_indices = [residue_indices[i] for i in N_save_order]

        for i in range(0,len(df_res)):
            df_res.at[i, 'indices_residue'] = [ordered_residue_indices[i]]

        return df_res, cyclization

    def create_substructure_sdf(self,df_res,df_mol,mol,cyclization,output_dir,pdb_name):
        if df_res.empty:
            return df_res, df_mol

        smiles_residues = []

        molecule_pdb_file = os.path.join(output_dir,'molecule.pdb')
        residue_pdb_file = os.path.join(output_dir,'residue.pdb')

        mol.write("pdb", molecule_pdb_file, overwrite=True)

        all_res_mol = []
        sequence = []
        residue_indices = [item[0] if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) else item
                   for item in df_res['indices_residue'].tolist()]
        check_sequence = df_res['check_sequence'][len(df_res)-1][0]
        unreasonable_bond_length = df_res['unreasonable_bond_length'][len(df_res)-1][0]
        NCAA_num_list = df_mol['NCAA_num'][len(df_mol)-1][0]
        NCAA_num = NCAA_num_list[0]
        unreasonable = {}

        for i in range(0,len(residue_indices)):
            residue_index = residue_indices[i]
            unreasonable_bond = None
            with open(residue_pdb_file,'w') as res_file:
                res_file.truncate(0)

            with open(molecule_pdb_file,'r') as molecule_file:
                for molecule_line in molecule_file:
                    if 'HETATM' in molecule_line or 'ATOM' in molecule_line:
                        pdb_index = round(float(molecule_line[6:11]))
                        if pdb_index in residue_index:
                            with open(residue_pdb_file,'a') as res_file:
                                res_file.write(molecule_line)
            with open(residue_pdb_file,'r') as res_file:
                for res_line in res_file:
                    if 'HETATM' in res_line or 'ATOM' in res_line:
                        res_name = res_line[17:20]
                        break

            # Now convert the residue.pdb to SMILES
            residue_mol = next(pybel.readfile('pdb', residue_pdb_file))
            all_res_mol.append(residue_mol)
            smiles = residue_mol.write("smi").strip()

            # Remove unwanted part of the string
            cleaned_smiles = smiles.split("\t")[0]
            if '.' in cleaned_smiles:
                unreasonable_bond = 'Structure Gaps'
            smiles_residues.append(cleaned_smiles)

            # Check residue index in unreasonable_bond_length_list
            for res in residue_index:
                for index, d in enumerate(unreasonable_bond_length):
                    if any(res in key for key in d.keys()):
                        unreasonable_bond = unreasonable_bond_length[index]
                        break

            if 'UN' not in res_name:
                sequence.append(res_name)
            else:
                res_name,NCAA_num = self.NCAA_check(cleaned_smiles,NCAA_num,check_sequence,residue_index)

                sequence.append(res_name)

            if unreasonable_bond is not None:
                unreasonable.update({f'{res_name}_{i}':unreasonable_bond})

        #Add row to df_mol
        new_row = {"Name": [[pdb_name]], "cyclization": [[cyclization]], "NCAA_num": [[NCAA_num]], "sequence": [sequence],
                   "unreasonable": [unreasonable], "residue_indices": residue_indices,
                   "atom_indices": df_res['atom_indices'][len(df_res)-1]}
        df_mol = pd.concat([df_mol, pd.DataFrame([new_row])], ignore_index=True)

        return df_res,df_mol

    def mol_to_sequence(self, output_dir, filter_unknown_aas=True, filter_non_cyclized=True, verbose=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            NCAA_num = self.Find_NCAA_num()
        except:
            NCAA_num = 1

        valid_mols = self.valid_mols
        name = f'sample_{valid_mols+1}.pdb'
        pdb_state_name = os.path.join(output_dir,name)

        df_mol = pd.DataFrame([{'Name': [[None]],'cyclization': [[None]],'NCAA_num': [[NCAA_num]],'sequence': [[None]],
                                    "unreasonable": [[None]],'residue_indices': [[None]], 'atom_indices': [[None]] }])

        df_res = pd.DataFrame(columns = ['atom_indices','indices_residue','CB_index','unreasonable_bond_length',
                                           'N_index','all_indices','check_sequence'])

        mol = self.mol
        unreasonable_bond_length = []
        check_sequence = {}
        all_indices = []

        state_pdb = 'molecule_state.pdb'
        with open(state_pdb, 'w') as writer:
            writer.write(mol.write("pdb"))
        atom_indices = {}
        res_num = 1

        smiles = mol.write("smi").strip()

        for atom in mol.atoms:
            all_indices.append(atom.idx)

            if atom.atomicnum == 6:  # Carbon only
                atom_CAname = atom_Nname = atom_Cname = atom_Oname = atom_CBname = atom_CMname = None

                # Use Open Babel for chirality
                if atom.OBAtom.IsChiral():
                    atom_CAname = f"CA:{res_num}"; CA_index = atom.idx

                    for bond in openbabel.OBAtomBondIter(atom.OBAtom):
                        neighbor_atom = bond.GetNbrAtom(atom.OBAtom)
                        if neighbor_atom.GetAtomicNum() == 7:
                            atom_Nname = f"N:{res_num}"; N_index = neighbor_atom.GetIdx()
                            for bondN in openbabel.OBAtomBondIter(neighbor_atom):
                                neighbor_atomN = bondN.GetNbrAtom(neighbor_atom)
                                if neighbor_atomN.GetAtomicNum() == 6 and bondN.GetBondOrder() == 1:
                                    bonded_elements = [b.GetNbrAtom(neighbor_atomN).GetAtomicNum() for b in openbabel.OBAtomBondIter(neighbor_atomN)]
                                    non_H_neighbors = [z for z in bonded_elements if z != 1]
                                    if len(non_H_neighbors) == 1 and non_H_neighbors[0] == 7:
                                        atom_CMname = f"CM:{res_num}"; CM_index = neighbor_atomN.GetIdx()

                        elif neighbor_atom.GetAtomicNum() == 6:
                            O_found = False
                            for bond2 in openbabel.OBAtomBondIter(neighbor_atom):
                                neighbor_atom2 = bond2.GetNbrAtom(neighbor_atom)
                                if O_found:
                                    break
                                if neighbor_atom2.GetAtomicNum() == 8 and bond2.GetBondOrder() == 2:
                                    O_found = True
                                    atom_Cname = f"C:{res_num}"; C_index = neighbor_atom.GetIdx()
                                    atom_Oname = f"O:{res_num}"; O_index = neighbor_atom2.GetIdx()
                            if not O_found:
                                atom_CBname = f"CB:{res_num}"; CB_index = neighbor_atom.GetIdx()

                    if all(name is not None for name in [atom_CAname, atom_Nname, atom_Cname, atom_Oname]):
                        indices_residue = []
                        atom_indices.update({atom_CAname: CA_index})
                        atom_indices.update({atom_Nname: N_index})
                        atom_indices.update({atom_Cname: C_index})
                        atom_indices.update({atom_Oname: O_index})
                        indices_residue.append(CA_index); indices_residue.append(N_index)
                        indices_residue.append(C_index); indices_residue.append(O_index)
                        if atom_CMname is not None:
                            atom_indices.update({atom_CMname: CM_index})
                            indices_residue.append(CM_index)
                        if atom_CBname is not None:
                            atom_indices.update({atom_CBname: CB_index})
                            indices_residue.append(CB_index)

                            new_row = {"atom_indices": [atom_indices], "indices_residue":[indices_residue], 'CB_index':[[CB_index]],
                                        'unreasonable_bond_length': [unreasonable_bond_length],'N_index': [[N_index]],'all_indices': [all_indices],
                                        'check_sequence': [check_sequence]}
                            df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
                            df_res = self.find_branching_atoms(df_res,mol)
                            check_sequence = df_res['check_sequence'][len(df_res)-1][0]
                            unreasonable_bond_length = df_res['unreasonable_bond_length'][len(df_res)-1][0]


                        res_num += 1
        if df_res.empty:
            new_row = {"atom_indices": [{}], "indices_residue":[[]], 'CB_index':[[]],'unreasonable_bond_length': [{}],
                'N_index': [[]],'all_indices': [all_indices], 'check_sequence': [{}]}
            df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)

        df_res = self.gly_residues(df_res,mol)
        df_res = self.check_N_methyl(df_res,mol)
        df_res,cyclization = self.order_residues(df_res,mol,smiles)
        df_res,df_mol = self.create_substructure_sdf(df_res,df_mol,mol,cyclization,output_dir,pdb_state_name)
        df_mol = df_mol[df_mol['Name'].astype(str) != '[[None]]'].reset_index(drop=True)

        if df_res.empty:
            return(None)

        ########  FILTER UNKNOWN AAs AND NON-CYCLIZED SEQUENCES #########

        # Start with all rows as True
        mask = pd.Series(True, index=df_mol.index)

        # Filter unknown amino acids if requested
        if filter_unknown_aas:
            unk_mask = ~df_mol['sequence'].apply(lambda seq_list: any(aa is not None and ("UK" in aa.upper() or "UNK" in aa.upper()) for seq in seq_list for aa in seq))
            mask &= unk_mask
            if verbose:
                print(f"{unk_mask.sum()} sequences kept; {len(df_mol) - unk_mask.sum()} removed due to unknown AAs")

        # Filter non-cyclized sequences if requested
        if filter_non_cyclized and 'cyclization' in df_mol.columns:
            cyclized_mask = df_mol['cyclization'].apply(lambda x: x is not None and any(subx is True for sublist in x for subx in sublist))
            mask &= cyclized_mask
            if verbose:
                print(f"{cyclized_mask.sum()} cyclized sequences kept; {len(df_mol) - cyclized_mask.sum()} removed due to non-cyclization")

        # Apply combined mask
        df_mol = df_mol[mask]

        # Return None if no sequences remain
        if df_mol.empty:
            if verbose:
                print("No sequences left after filtering. Returning None.")
            return None

        state_pdb_files  = self.seq_to_pdb(df_mol,output_dir)
        df_mol = self.reorder_combine_pdb(df_mol,state_pdb_files, verbose=verbose)

        ######################################
        return df_mol
