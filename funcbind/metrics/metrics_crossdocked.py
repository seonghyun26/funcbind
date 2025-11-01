from posecheck import PoseCheck
from rdkit import Chem, DataStructs
from rdkit.DataManip.Metric import GetTanimotoDistMat
import torch
import os
import numpy as np
from copy import deepcopy

from funcbind.utils.evaluation.docking_qvina import QVinaDockingTask
from funcbind.utils.evaluation.docking_vina import VinaDockingTask
from funcbind.utils.evaluation.scoring_func import get_chem
from funcbind.metrics.metrics import MetricsSampling


class MetricsSamplingCrossDocked(MetricsSampling):
    def __init__(self, config, target_dirname=None):
        super().__init__(config, target_dirname)
        self.docking = True
        self.vina = config["sampling"]["docking_mode"] in ["vina_score", "vina_full", "vina_dock"]
        self.posecheck = config["sampling"]["posecheck"]
        self.res_dock = []
        self.res_posecheck = []
        self.config = config

    def update(self, ligand_gt: dict = None, receptor: dict = None, seq_gen = None, attempts=None, **kwargs):
        super().update()
        if attempts is None:
            attempts = self.config["sampling"]["n_attempts"]
        self.molecular_stats[0][0]['validity'] = round(len(seq_gen) / (self.config["sampling"]["n_chains"] * attempts), 2) if seq_gen is not None else 0

        # update dock and chem metrics
        if self.docking:
            try:
                target_res = self.eval_target_dock(ligand_gt, receptor)
                self.res_dock.append(target_res)
            except Exception as e:
                print(f">> FAILED TO COMPUTE DOCKING: {e}")

        # update posecheck (strain and clashes)
        if self.posecheck:
            try:
                target_res = self.eval_target_posecheck(receptor)
                self.res_posecheck.append(target_res)
            except Exception as e:
                print(f">> FAILED TO COMPUTE POSECHECK: {e}")

    def save(self, name="metrics.pt"):
        torch.save(
            {"molecular_stats": self.molecular_stats, "res_dock": self.res_dock, "res_posecheck": self.res_posecheck},
            os.path.join(self.target_dirname, name)
        )

    def reset(self):
        super().reset()
        self.res_dock = []
        self.res_posecheck = []

    def compute(self, docking_mode: str, remove_vina_outliers: bool = True):
        if len(self.molecular_stats) == 0 and len(self.res_dock) == 0 and len(self.res_posecheck) == 0:
            return None
        out = super().compute()
        out['validity'] = [r[0]['validity'] for r in self.molecular_stats]

        # docking
        results = self.res_dock
        if self.docking:
            if docking_mode in ["vina", "qvina"]:
                out["vina"] = self._extract_vina_metric(results, ["vina", 0, "affinity"])
            elif docking_mode in ["vina_full", "vina_score"]:
                if remove_vina_outliers:
                    out["vina_score"] = self._extract_vina_metric(results, ["vina", "score_only", 0, "affinity"], max_threshold=15)
                    out["vina_min"] = self._extract_vina_metric(results, ["vina", "minimize", 0, "affinity"], max_threshold=15)
                else:
                    out["vina_score"] = self._extract_vina_metric(results, ["vina", "score_only", 0, "affinity"])
                    out["vina_min"] = self._extract_vina_metric(results, ["vina", "minimize", 0, "affinity"])
            if docking_mode in ["vina_full", "vina_dock"]:
                out["vina_dock"] = self._extract_vina_metric(results, ["vina", "dock", 0, "affinity"])

            # chem metrics
            try:
                out["qed"] = [x["chem_results"]["qed"] for r in results for x in r]
            except Exception:
                out["qed"] = [np.nan]
            try:
                out["sa"] = [x["chem_results"]["sa"] for r in results for x in r]
            except Exception:
                out["sa"] = [np.nan]

            out["diversity"] = self._extract_diversity_metric(results)

        # posecheck
        results = self.res_posecheck
        if self.posecheck and results is not None:
            out["clashes"] = self._extract_posecheck_metric(results, "clashes")
            out["strain"] = self._extract_posecheck_metric(results, "strain")

        return out

    def _extract_vina_metric(self, results, path, max_threshold=None):
        """Extract Vina metrics with optional filtering."""
        try:
            values = []
            for r in results:
                for x in r:
                    nested_val = x
                    for key in path:
                        if nested_val is None:
                            break
                        nested_val = nested_val[key]

                    if nested_val is not None:
                        if max_threshold is None or nested_val <= max_threshold:
                            values.append(nested_val)
            return values if values else [np.nan]
        except Exception:
            return [np.nan]

    def _extract_diversity_metric(self, results):
        """Extract diversity metric."""
        try:
            return [x for r in results for x in r[0]["diversity"]]
        except Exception:
            return [np.nan]

    def _extract_posecheck_metric(self, results, metric_name):
        """Extract posecheck metrics with filtering."""
        try:
            return [x for r in results for x in r[metric_name] if x is not None and not np.isnan(x)]
        except Exception:
            return [np.nan]

    def eval_target_posecheck(self, pocket: dict = None):
        """
        Evaluate the target pose check metrics.

        Args:
            target_dirname (str): The directory name of the target.
            pocket (dict): The pocket information.
            cfg (DictConfig): The configuration.

        Returns:
            dict: The target pose check metrics including clashes and strain energy.
        """
        pc = PoseCheck()
        try:
            gen_ligands_fn = os.path.join(self.target_dirname, "samples.sdf")
            pc.load_ligands_from_sdf(gen_ligands_fn)
            target_res = {"strain": pc.calculate_strain_energy()}
        except Exception as e:
            print(f">> FAILED TO COMPUTE STRAIN: {e}")
            target_res = {"strain": [np.nan]}
        if pocket is not None:
            try:
                protein_fn = os.path.join(os.path.dirname(pocket["id"][0]), os.path.basename(pocket["id"][0])[:10] + ".pdb")
                protein_path = os.path.join(self.config["dset"]["data_dir"], "crossdocked_v1.1_rmsd1.0", protein_fn)
                pc.load_protein_from_pdb(protein_path)
                target_res.update({"clashes": pc.calculate_clashes()})
            except Exception as e:
                print(f">> FAILED TO COMPUTE CLASHES: {e}")
                target_res.update({"clashes": [np.nan]})
        else:
            target_res.update({"clashes": [np.nan]})
        return target_res


    def eval_target_dock(self, ligand_gt: dict = None, pocket: dict = None):
        """
        Evaluate docking for a target using the provided ligand and pocket.

        Args:
            target_dirname (str): The directory path of the target.
            ligand_gt (dict): The ground truth ligand information.
            pocket (dict): The pocket information.
            cfg (DictConfig): The configuration object.

        Returns:
            list: A list of dictionaries containing the evaluation results for each ligand.

        """
        sdfs_path = os.path.join(self.target_dirname, "samples.sdf")
        if not os.path.exists(sdfs_path):
            return []
        mols = Chem.SDMolSupplier(sdfs_path)

        try:
            fingerprints = [Chem.RDKFingerprint(mol) for mol in mols if mol is not None]
            distmat = GetTanimotoDistMat(fingerprints)
        except Exception:
            distmat = [0]

        res = []
        for idx, mol in enumerate(mols):
            if mol is None:
                continue
            coords, atom_types = [], []
            for i, atom in enumerate(mol.GetAtoms()):
                positions = mol.GetConformer().GetAtomPosition(i)
                coords.append([positions.x, positions.y, positions.z])
                atom_types.append(atom.GetAtomicNum())
            coords = np.array(coords)

            lig = {
                "mol": mol,
                "smiles": Chem.MolToSmiles(mol),
                "chem_results": get_chem(mol),
                "diversity": distmat,  # this is already contains all pairwise distances
                "pred_pos": coords,
                "pred_v": atom_types,
            }
            if ligand_gt is not None and pocket is not None and self.vina:
                ligand_gt_fn, pocket_fn = ligand_gt["id"][0], pocket["id"][0]
                protein_root = os.path.join(self.config["dset"]["data_dir"], "crossdocked_v1.1_rmsd1.0")
                try:
                    res_dock = dock(mol, ligand_gt_fn, protein_root, center=pocket["center_coords"].squeeze().cpu().numpy().astype(float), docking_mode=self.config["sampling"]["docking_mode"], exhaustiveness=self.config["sampling"]["exhaustiveness"])
                except Exception as e:
                    print(f">> FAILED TO DOCK INDEX {idx}: {e}")
                    res_dock = {"score_only": np.nan, "minimize": np.nan, "dock": np.nan}
                lig.update({
                    "ligand_filename": ligand_gt_fn,
                    "pocket_filename": pocket_fn,
                    "vina": res_dock,
                })
            res.append(lig)
        return res

    def log_results(self, fabric, t0, docking_mode, receptor_id=None, n_atoms_gt=None):
        res_log, res = super().log_results(fabric, t0, docking_mode, receptor_id, n_atoms_gt)

        # Log validity metric
        self._log_simple_metric(res, res_log, fabric, "validity", "Validity")

        # Docking metrics
        if self.docking:
            # Chemical metrics with mean and median
            chem_metrics = [
                ("qed", "QED"),
                ("sa", "SA"),
                ("diversity", "Diversity")
            ]

            for metric_name, display_name in chem_metrics:
                self._log_metric_with_mean_median(res, res_log, fabric, metric_name, display_name)

            # Vina metrics based on docking mode
            if docking_mode != "none" and docking_mode != "vina_dock":
                vina_metrics = [
                    ("vina_score", "Vina_score"),
                    ("vina_min", "Vina_min")
                ]
                for metric_name, display_name in vina_metrics:
                    self._log_metric_with_mean_median(res, res_log, fabric, metric_name, display_name)

            if docking_mode in ["vina_full", "vina_dock"]:
                self._log_metric_with_mean_median(res, res_log, fabric, "vina_dock", "Vina_dock")

        # Pose check metrics
        if self.posecheck:
            posecheck_metrics = [
                ("clashes", "Clashes"),
                ("strain", "Strain")
            ]
            for metric_name, display_name in posecheck_metrics:
                self._log_metric_with_mean_median(res, res_log, fabric, metric_name, display_name)

        return res_log, res

    def _log_metric_with_mean_median(self, res, res_log, fabric, metric_name, display_name):
        """Log a metric with both mean and median values."""
        if metric_name in res:
            mean_val = np.nanmean(res[metric_name])
            median_val = np.nanmedian(res[metric_name])
            if not np.isnan(mean_val):
                res_log.update({
                    f"{metric_name}_mean": mean_val,
                    f"{metric_name}_median": median_val
                })
                fabric.print(f"{display_name}: Mean: {mean_val:.3f}  Median: {median_val:.3f}")

def dock(
    mol: Chem.Mol,
    ligand_gt_fn: str,
    protein_root: str,
    center: list = None,
    docking_mode: str = "none",
    exhaustiveness: int = 32
):
    """
    Perform docking of a molecule to a protein using various docking modes.

    Args:
        mol (Chem.Mol): The molecule to be docked.
        ligand_gt_fn (str): The filename of the ligand ground truth.
        protein_root (str): The root directory of the protein files.
        center (list, optional): The center coordinates for docking. Defaults to None.
        docking_mode (str, optional): The docking mode to be used. Defaults to "none".
        exhaustiveness (int, optional): The exhaustiveness parameter for docking. Defaults to 32.

    Returns:
        dict or None: The docking results, or None if docking_mode is "none".
    """
    if docking_mode == "qvina":
        vina_task = QVinaDockingTask.from_generated_mol(
            mol, ligand_gt_fn, protein_root=protein_root, center=2
        )
        vina_results = vina_task.run_sync()
    elif docking_mode == "vina":
        vina_task = VinaDockingTask.from_generated_mol(
            mol, ligand_gt_fn, protein_root=protein_root
        )
        vina_results = vina_task.run(mode="dock")
    elif docking_mode in ["vina_full", "vina_score", "vina_dock"]:
        vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol), ligand_gt_fn, protein_root=protein_root, center=center)
        vina_results = {
            "score_only": vina_task.run(mode="score_only", exhaustiveness=exhaustiveness),
            "minimize": vina_task.run(mode="minimize", exhaustiveness=exhaustiveness)
        }
        if docking_mode in ["vina_full", "vina_dock"]:
            vina_results.update({
                "dock": vina_task.run(mode="dock", exhaustiveness=exhaustiveness),
            })
    elif docking_mode == "none":
        vina_results = None
    else:
        raise NotImplementedError

    return vina_results


def get_diversity(
    fp_ref: DataStructs.ExplicitBitVect,
    fingerprints: list
):
    """
    Calculate the diversity of a set of fingerprints compared to a reference fingerprint.

    Parameters:
        fp_ref (DataStructs.ExplicitBitVect): The reference fingerprint.
        fingerprints (list): A list of fingerprints to compare.

    Returns:
        float: The average diversity score.

    """
    divs = [1 - DataStructs.TanimotoSimilarity(fp, fp_ref) for fp in fingerprints]
    return np.array(divs).mean()