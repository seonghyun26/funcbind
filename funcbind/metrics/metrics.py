from rdkit import Chem
import torch
import torchmetrics
import os
import numpy as np
import time
# import warnings
from collections import Counter
import matplotlib.pyplot as plt


class MetricsSampling(torchmetrics.MetricCollection):
    def __init__(self, config, target_dirname=None):
        """
        Metrics for sampling.

        Args:
            docking (bool): Flag indicating whether to compute docking metrics. Default is True.
            posecheck (bool): Flag indicating whether to compute posecheck metrics. Default is True.

        Attributes:
            docking (bool): Flag indicating whether to compute docking metrics.
            posecheck (bool): Flag indicating whether to compute posecheck metrics.
            res_dock (list): List to store docking results.
            res_posecheck (list): List to store posecheck results.
        """
        self.molecular_stats = []
        self.target_dirname = target_dirname if target_dirname is not None else config["dirname"]
        self.docking = False
        self.posecheck = False
        self.general = False
        self.segment_reconstruction = False
        self.mcpp_metrics = False
        self.config = config

    def update(self, **kwargs):
        """
        Update the metrics with new results.

        Args:
            target_dirname (str): Directory name of the target.
            ligand_gt (dict): Ground truth ligand.
            pocket (dict): Pocket.
            cfg (DictConfig): Configuration parameters.
        """
        # molecular stats
        try:
            molecular_stats = self.eval_molecular_stats()
        except Exception:
            molecular_stats = []
        self.molecular_stats.append(molecular_stats)

    def save(self, name="metrics.pt"):
        """
        Save the metrics results.
        """
        torch.save(
            {"molecular_stats": self.molecular_stats},
            os.path.join(self.target_dirname, name)
        )

    def reset(self):
        """
        Reset the metrics.
        """
        self.molecular_stats = []

    def eval_molecular_stats(self):
        """
        Evaluate molecular stats.
        Returns:
            list: A list of dictionaries containing the evaluation results for each ligand.

        """
        sdfs_path = os.path.join(self.target_dirname, "samples.sdf")
        if not os.path.exists(sdfs_path):
            return []
        mols = Chem.SDMolSupplier(sdfs_path)

        res = []
        for mol in mols:
            if mol is None:
                continue
            lig = {
                "n_atoms": mol.GetNumAtoms(),
                "n_fragments": len(Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
            }
            res.append(lig)
        return res


    def compute(self, **kwargs):
        """
        Compute the metrics.

        Args:
            docking_mode (str): Docking mode.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if len(self.molecular_stats) == 0:
            return None

        out = {}

        # molecular stats
        results = self.molecular_stats
        try:
            n_atoms = [x["n_atoms"] for r in results for x in r]
        except Exception:
            n_atoms = [np.nan]
        try:
            n_fragments = [x["n_fragments"] for r in results for x in r]
        except Exception:
            n_fragments = [np.nan]
        out.update({
            "n_atoms": n_atoms,
            "n_fragments": n_fragments
        })

        return out


    def log_results(self, fabric, t0, docking_mode, receptor_id=None, n_atoms_gt=None):
        if receptor_id is not None:
            fabric.print("================================================")
            fabric.print(f"receptor {receptor_id} - {n_atoms_gt} atms - {(time.time() - t0):.2f}s")
        else:
            fabric.print("################################################")
            fabric.print("final results")
        res = self.compute(docking_mode=docking_mode)

        # molecular stats
        res_log = {}
        try:
            res_log.update({
                "n_atoms_mean": np.mean(res["n_atoms"]) if res["n_atoms"] is not None else 0,
                "n_atoms_median": np.median(res["n_atoms"]) if res["n_atoms"] is not None else 0,
                "frac_single_frag": sum([1 for n in res["n_fragments"] if n == 1]) / len(res["n_fragments"]) if len(res["n_fragments"]) > 0 else 0,
            })
            fabric.print(f"Single fragment : {res_log['frac_single_frag']:.3f}")
            fabric.print(f"N atoms:    Mean: {res_log['n_atoms_mean']:.3f}  Median: {res_log['n_atoms_median']:.3f} v.s. GT: {n_atoms_gt}")
        except:
            fabric.print(f"res_log.update failed")

        return res_log, res

    def _log_simple_metric(self, res, res_log, fabric, metric_name, display_name=None, precision=3):
        """Log a simple metric with mean value using NaN-safe calculation."""
        if metric_name in res:
            mean_val = np.nanmean(res[metric_name])
            if not np.isnan(mean_val):
                res_log.update({metric_name: mean_val})
                display = display_name or metric_name.replace('_', ' ').title()
                fabric.print(f"{display}: {mean_val:.{precision}f}")

    def _log_metric_with_stats(self, res, res_log, fabric, metric_name, display_name=None, precision=3):
        """Log a metric with mean, max, and min statistics using NaN-safe calculations."""
        if metric_name in res:
            values = res[metric_name]
            mean_val = np.nanmean(values)
            max_val = np.nanmax(values)
            min_val = np.nanmin(values)
            if not np.isnan(mean_val):
                res_log.update({metric_name: mean_val})
                display = display_name or metric_name.replace('_', ' ').title()
                fabric.print(f"{display}: Mean: {mean_val:.{precision}f}  Max: {max_val:.{precision}f}  Min: {min_val:.{precision}f}")

    def _log_percentage_metric(self, res, fabric, metric_name, display_name=None):
        """Log a percentage metric (e.g., percentage of values < 0)."""
        if metric_name in res:
            percentage = sum(1 for val in res[metric_name] if val < 0) / len(res[metric_name])
            display = display_name or f"Percentage of {metric_name} < 0"
            fabric.print(f"{display}: {percentage:.4f}")

    def _create_sequence_histogram(self, res):
        """Create and save a histogram of sequence lengths."""
        if len(res['seq_seed_len']) == 1:
            try:
                len_counter_sorted = sorted(sum(res['seq_len_counter'], Counter()).items())
                sequence_lengths, counts = zip(*len_counter_sorted)
                plt.bar(sequence_lengths, counts, color='blue')
                bars = plt.bar(sequence_lengths, counts, color='blue')
                for bar, x in zip(bars, sequence_lengths):
                    if x == res['seq_seed_len'][0]:
                        bar.set_color('red')
                plt.xlabel('Generated Sequence Length')
                plt.ylabel('Count')
                plt.title('Histogram of Generated Sequence Lengths')
                plt.savefig(os.path.join(self.target_dirname, "seq_len_hist.png"))
                plt.close()
            except Exception as e:
                print(f"Failed to plot histogram of sequence lengths: {e}")

    def _log_computed_stats(self, res, res_log, fabric, mean_key, max_key, min_key, display_name, precision=3):
        """Log pre-computed mean/max/min statistics in a single line."""
        if all(key in res for key in [mean_key, max_key, min_key]):
            mean_val = np.nanmean(res[mean_key])
            max_val = np.nanmean(res[max_key])
            min_val = np.nanmean(res[min_key])
            if not np.isnan(mean_val):
                res_log.update({mean_key: mean_val})
                fabric.print(f"{display_name}: Mean: {mean_val:.{precision}f}  Max: {max_val:.{precision}f}  Min: {min_val:.{precision}f}")

    def _safe_extract_metric(self, results, metric_name, default_value=np.nan):
        """
        Safely extract a metric from results with proper error handling.

        Args:
            results: List of result dictionaries
            metric_name: Name of the metric to extract
            default_value: Value to return if extraction fails

        Returns:
            List of metric values or default_value list if extraction fails
        """
        try:
            return [x[metric_name] for r in results for x in r]
        except (KeyError, TypeError, AttributeError) as e:
            # warnings.warn(f"Failed to extract metric '{metric_name}': {e}")
            return [default_value]
        except Exception as e:
            # warnings.warn(f"Unexpected error extracting metric '{metric_name}': {e}")
            return [default_value]