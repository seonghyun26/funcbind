from functools import partial
import warnings
import numpy as np
from funcbind.metrics.metrics import MetricsSampling
from funcbind.utils.utils_rosetta import interface_energy
from funcbind.dataset.preprocess_sabdab import get_numbering
import os
import pandas as pd
from glob import glob
from collections import Counter
from tqdm import tqdm
import torch
from cpdb import parse
from multiprocessing import Pool, cpu_count
from biopandas.pdb.engines import amino3to1dict


class MetricsSamplingAb(MetricsSampling):
    def __init__(self, config, target_dirname=None, library_binders=None):
        super().__init__(config, target_dirname)
        self.general_stats = []
        self.segment_reconstruction_scores = []
        self.general = True
        self.segment_reconstruction = True


    def update(self, ligand_gt: dict = None, receptor: dict = None, seq_gen = None, seq_seed = None, attempts=None, unique_only=False, **kwargs):
        from funcbind.utils.utils_metrics import extract_sequences_from_pdb

        # Samples
        self.seq_to_path = {}
        samples_path = os.path.join(self.target_dirname, "samples.pdb")
        seq_gen = extract_sequences_from_pdb(samples_path, seq_to_path=self.seq_to_path, target_dirname=self.target_dirname, ligand_id=ligand_gt['id'][0] if ligand_gt is not None else None)

        if attempts is None:
            attempts = self.config["sampling"]["n_attempts"]

        if self.general:
            print("------------------------General stats------------------------")
            super().update()
            general_stats, seq_seed_length = self.eval_general_stats(seq_gen, seq_seed, verbose=False)
            general_stats[0]['validity'] = round(len(seq_gen) / (self.config["sampling"]["n_chains"] * attempts), 2)
            self.general_stats.append(general_stats)
        if self.segment_reconstruction:
            try:
                print("------------------------Segment stats------------------------")
                if unique_only:
                    pdb_files = [self.seq_to_path[seq][0] for seq in set(seq_seed_length)]
                else:
                    pdb_files = [pdb_file for seq in seq_seed_length for pdb_file in self.seq_to_path[seq]]
                segment_reconstruction_scores = self.eval_segment_reconstruction_scores(ligand_gt, receptor, segment=seq_seed, seq_alignment_scheme='aho', verbose=True, pdb_files=pdb_files)
            except Exception as e:
                segment_reconstruction_scores = [{"segment_seq_recovery": 0.0, "segment_rmsd": 0.0, "ddG": 0.0}]
            self.segment_reconstruction_scores.append(segment_reconstruction_scores)

    def reset(self):
        super().reset()
        self.general_stats = []
        self.segment_reconstruction_scores = []

    def eval_general_stats(self, seq_gen, seq_seed = None, verbose = True):
        # Uniqueness
        seq_set = set(seq_gen)
        seq_unique = list(seq_set)

        # Designs of size of the seed
        length_set = {len(seq) for seq in seq_unique}
        grouped_by_length = {length: [seq for seq in seq_unique if len(seq) == length] for length in length_set}
        seq_seed_length = grouped_by_length.get(len(seq_seed), [])

        # Length stats
        seq_len = [len(seq) for seq in seq_unique]
        seq_len_counter = Counter(seq_len)

        res = []
        lig = {
            "uniqueness": len(seq_set) / len(seq_gen) if len(seq_gen) > 0 else 0,
            "seq_len_mean": np.mean(seq_len) if len(seq_len) > 0 else 0,
            "seq_len_min": np.min(seq_len) if len(seq_len) > 0 else 0,
            "seq_len_max": np.max(seq_len) if len(seq_len) > 0 else 0,
            "seq_len_counter": seq_len_counter,
        }
        if seq_seed is not None:
            lig.update({
                "seq_seed_len": len(seq_seed),
                "ratio_with_seed_len": seq_len_counter[len(seq_seed)] / len(seq_unique) if len(seq_unique) > 0 else 0,
            })

        res.append(lig)
        return res, seq_seed_length


    def process_for_reconstruction(self, pdb_path, seq_alignment_scheme):
         # Load reference info
        df = parse(pdb_path, df=True)
        # Only keep CA atoms
        df = df[df['atom_name'] == 'CA']
        # Get reference sequence
        H_seq = "".join([amino3to1dict.get(s.strip(), 'X') for s in df[df['chain_id'] == 'H']['residue_name']])
        try:
            L_seq = "".join([amino3to1dict.get(s.strip(), 'X') for s in df[df['chain_id'] == 'L']['residue_name']])
            seq = H_seq + L_seq
            # Get aho alignment
            df['aho_num'] = np.concatenate((get_numbering(H_seq, scheme=seq_alignment_scheme).numpy(), 149 + get_numbering(L_seq, scheme=seq_alignment_scheme).numpy()))
            interface_str = 'HL_A'
        except Exception as e:  # no L chain
            seq = H_seq
            # Get aho alignment
            df['aho_num'] = get_numbering(H_seq, scheme=seq_alignment_scheme).numpy()
            interface_str = 'H_A'
        return seq, df, interface_str

    def eval_segment_reconstruction_scores(self, ligand_gt: dict, receptor: dict, segment: str, seq_alignment_scheme: str = 'aho', verbose: bool = True, pdb_files = None) -> float:
        # Build paths
        if pdb_files is None:
            reference_pdb_path = os.path.join(self.target_dirname, f"{ligand_gt['id'][0]}.pdb") if ligand_gt is not None else None
        else:
            reference_pdb_path = os.path.join(os.path.dirname(pdb_files[0]), f"{ligand_gt['id'][0]}.pdb") if ligand_gt is not None else None
        reference_receptor_pdb_path = os.path.join(self.target_dirname, f"{receptor['id'][0]}.pdb")
        ref_seq, ref_df, interface_str = self.process_for_reconstruction(reference_pdb_path, seq_alignment_scheme)
        # Get segment indices
        ref_segment_start = ref_seq.find(segment)
        ref_segment_end = ref_segment_start + len(segment)
        ref_df_segment = ref_df.iloc[ref_segment_start:ref_segment_end]
        # Get reference energy (old DiffAb)
        ref_dG = interface_energy(reference_pdb_path, reference_receptor_pdb_path, interface=interface_str)
        res = []
        if pdb_files is None:
            pdb_files = glob(f"{self.target_dirname}/*_full_ab_*.pdb")
            sorted_pdb_files = sorted(pdb_files, key=extract_number)
        else:
            sorted_pdb_files = pdb_files

        process_partial = partial(
            self.segment_reconstruction_score_per_pdb,
            reference_receptor_pdb_path=reference_receptor_pdb_path,
            ref_seq=ref_seq,
            ref_df_segment=ref_df_segment,
            ref_segment_start=ref_segment_start,
            ref_segment_end=ref_segment_end,
            ref_dG=ref_dG,
            seq_alignment_scheme=seq_alignment_scheme,
            verbose=verbose
        )
        with Pool(processes=cpu_count()) as pool:
            res = list(
                tqdm(pool.imap(process_partial, sorted_pdb_files),
                    total=len(sorted_pdb_files),
                    desc="Processing PDB files")
            )
        return res

    def segment_reconstruction_score_per_pdb(self, pdb_path, reference_receptor_pdb_path, ref_seq, ref_df_segment, ref_segment_start, ref_segment_end, ref_dG, seq_alignment_scheme, verbose = True):
        seq, df, interface_str = self.process_for_reconstruction(pdb_path, seq_alignment_scheme)
        # Get segment indices
        segment_start = ref_segment_start
        segment_end = seq.find(ref_seq[ref_segment_end:])
        df_segment = df.iloc[segment_start:segment_end]
        # Join segment dataframes by aho number
        df_segment = pd.merge(df_segment, ref_df_segment, on=['aho_num'], suffixes=('', '_ref'), how="outer")

        # Compute amino acid recovery
        recovery = np.mean(df_segment['residue_name'] == df_segment['residue_name_ref'])
        # Compute RMSD
        rmsd = np.sqrt(np.nanmean((df_segment['x_coord'] - df_segment['x_coord_ref'])**2 + (df_segment['y_coord'] - df_segment['y_coord_ref'])**2 + (df_segment['z_coord'] - df_segment['z_coord_ref'])**2))
        # Compute interface energy
        dG = interface_energy(pdb_path, reference_receptor_pdb_path, interface=interface_str)
        ddG = dG - ref_dG
        if verbose:
            # debug here
            seq_gen = "".join([amino3to1dict.get(s, '-') for s in df_segment['residue_name']])
            if ddG <= 0:
                print(f"PDB: {os.path.join(*pdb_path.split('/')[-4:])}, Seq: {seq_gen}, Recovery: {recovery:.3f}, RMSD: {rmsd:.3f}, ddG: {ddG:.3f}")
        return {
            "segment_seq_recovery": recovery,
            "segment_rmsd": rmsd,
            "ddG": ddG,
        }


    def save(self, name="metrics.pt"):
        torch.save(
            {"molecular_stats": self.molecular_stats, "segment_reconstruction_scores": self.segment_reconstruction_scores, "general_stats": self.general_stats},
            os.path.join(self.target_dirname, name)
        )


    def compute(self, **kwargs):
        """
        Compute the metrics.

        Args:
            docking_mode (str): Docking mode.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if len(self.molecular_stats) == 0 and len(self.general_stats) == 0 and len(self.segment_reconstruction_scores) == 0:
            return None
        out = super().compute()

        # general_stats stats
        results = self.general_stats
        if len(results) > 0:
            out.update({
                "validity": self._safe_extract_metric(results, "validity"),
                "uniqueness": self._safe_extract_metric(results, "uniqueness"),
                "seq_len_mean": self._safe_extract_metric(results, "seq_len_mean"),
                "seq_len_min": self._safe_extract_metric(results, "seq_len_min"),
                "seq_len_max": self._safe_extract_metric(results, "seq_len_max"),
                "seq_seed_len": self._safe_extract_metric(results, "seq_seed_len"),
                "seq_len_counter": self._safe_extract_metric(results, "seq_len_counter", default_value=Counter()),
                "ratio_with_seed_len": self._safe_extract_metric(results, "ratio_with_seed_len"),
            })

        # Segment reconstruction (CDR H3) stats for antibody complexes
        results = self.segment_reconstruction_scores
        if len(results) > 0:
            out.update({
                "segment_seq_recovery": self._safe_extract_metric(results, "segment_seq_recovery"),
                "segment_rmsd": self._safe_extract_metric(results, "segment_rmsd"),
                "ddG": self._safe_extract_metric(results, "ddG"),
            })
        return out


    def log_results(self, fabric, t0, docking_mode, receptor_id=None, n_atoms_gt=None):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res_log, res = {}, {}

        if self.general:
            fabric.print("---------------- All designs results----------------")
            res_log, res = super().log_results(fabric, t0, docking_mode, receptor_id, n_atoms_gt)
            self._log_simple_metric(res, res_log, fabric, "validity", "Validity")
            self._log_simple_metric(res, res_log, fabric, "uniqueness", "Uniqueness")
            self._log_computed_stats(res, res_log, fabric, "seq_len_mean", "seq_len_max", "seq_len_min", "Seq len")
            self._log_simple_metric(res, res_log, fabric, "seq_seed_len", "Seq seed len mean")
            len_counter_sorted = sorted(sum(res['seq_len_counter'], Counter()).items())
            fabric.print(f"Seq len counter: {len_counter_sorted}")
            self._create_sequence_histogram(res)
            self._log_simple_metric(res, res_log, fabric, "ratio_with_seed_len", "Ratio sequences with seed's length")

        if self.general:
            fabric.print("---------------- Designs with seed's length results----------------")

        if self.segment_reconstruction:
            self._log_metric_with_stats(res, res_log, fabric, "segment_seq_recovery", "Segment seq recovery")
            self._log_metric_with_stats(res, res_log, fabric, "segment_rmsd", "Segment RMSD")
            self._log_simple_metric(res, res_log, fabric, "ddG", "ddG")
            if "ddG" in res:
                imp_percentage = sum(1 for ddG in res['ddG'] if ddG < 0) / len(res['ddG'])
                res_log.update({"IMP": imp_percentage})
                fabric.print(f"IMP: {imp_percentage:.4f}")

        return res_log, res


# Function to extract the numeric part from the filename
def extract_number(filename):
    # Assuming the number is between '_full_ab_' and '.pdb'
    base_name = os.path.basename(filename)
    number_part = base_name.split('_full_ab_')[1].split('.')[0]
    return int(number_part)
