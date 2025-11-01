# from https://github.com/guanjq/targetdiff/blob/main/utils/evaluation/docking_vina.py
import numpy as np
from openbabel import pybel
from meeko import MoleculePreparation
from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib
import getpass as gt
import meeko
import subprocess, sys, tempfile, os, pickle
from openbabel import openbabel as ob
import random
from datetime import datetime

from funcbind.utils.reconstruct import reconstruct_from_generated
from funcbind.utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


def safe_dock(ligand_pdbqt, protein_pdbqt, pocket_center, size_x, size_y, size_z, mode='dock', exhaustiveness=8, **kwargs):
    # Create temp script that will recreate the dock object
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = "-" + str(random.randint(0, 999999999999)).zfill(4)
    timestamp = timestamp_prefix + random_suffix
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write(f"""
import pickle, sys
from funcbind.utils.evaluation.docking_vina import VinaDock
try:
    # Recreate the dock object
    dock = VinaDock('{ligand_pdbqt}', '{protein_pdbqt}')
    dock.pocket_center = ({pocket_center[0]}, {pocket_center[1]}, {pocket_center[2]})
    dock.box_size = [{size_x}, {size_y}, {size_z}]
    # Perform docking
    score, pose = dock.dock(score_func='vina', mode="{mode}", exhaustiveness={exhaustiveness}, save_pose=True)
    print(f"SCORE:{{score}}")
    # Save pose to file if it exists
    if pose is not None:
        with open(f'temp_{timestamp}.pkl', 'wb') as f:
            pickle.dump(pose, f)
    sys.exit(0)
except Exception as e:
    print(f"ERROR:{{e}}")
    sys.exit(1)
""")
        script_path = f.name
    try:
        # Run subprocess
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        # Process result
        if result.returncode != 0:
            print(f"Docking failed: {result.stderr}")
            return np.nan, np.nan
        # Get score from output
        score = np.nan
        for line in result.stdout.split('\n'):
            if line.startswith("SCORE:"):
                score = float(line.replace("SCORE:", ""))
        # Get pose from file if exists
        pose = np.nan
        if os.path.exists(f'temp_{timestamp}.pkl'):
            with open(f'temp_{timestamp}.pkl', 'rb') as f:
                pose = pickle.load(f)
            os.remove(f'temp_{timestamp}.pkl')
        return score, pose
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
        if os.path.exists(f'temp_{timestamp}.pkl'):
            os.remove(f'temp_{timestamp}.pkl')


# from meeko/utils/obutils.py (removed in https://github.com/forlilab/Meeko/pull/157)
def writeMolecule(mol, fname=None, ftype=None):
    """save a molecule with openbabel"""
    if ftype is None:
        n, ftype = meeko.utils.getNameExt(fname)
        ftype = ftype.lower()

    conv = ob.OBConversion()
    conv.SetOutFormat(ftype)

    if not fname is None:
        conv.WriteFile(mol, fname)
    else:
        return conv.WriteString(mol)


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')

    def addH(self, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()


class PrepProt(object):
    def __init__(self, pdb_file):
        self.prot = pdb_file

    def del_water(self, dry_pdb_file):  # optional
        with open(self.prot) as f:
            lines = [li for li in f.readlines() if li.startswith('ATOM') or li.startswith('HETATM')]
            dry_lines = [li for li in lines if 'HOH' not in li]

        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30', '--ff=AMBER', self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object):
    def __init__(self, lig_pdbqt, prot_pdbqt):
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt

    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f:
            lines = [li for li in f.readlines() if li.startswith('ATOM') or li.startswith('HEATATM')]
            xs = [float(li[31:39]) for li in lines]
            ys = [float(li[39:47]) for li in lines]
            zs = [float(li[47:55]) for li in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size

    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket.
        buffer: buffer size to add

        if ref is not None:
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension
        else:
            use the entire protein to define pocket
        '''
        if ref is None:
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):
        # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only':
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError

        if not save_pose:
            return score
        else:
            if mode == 'score_only':
                pose = None
            elif mode == 'minimize':
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f:
                    v.write_pose(tmp.name, overwrite=True)
                with open(tmp.name, 'r') as f:
                    pose = f.read()

            elif mode == 'dock':
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())

        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(os.path.dirname(data.ligand_filename), os.path.basename(data.ligand_filename)[:10] + '.pdb')
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(os.path.dirname(ligand_filename), os.path.basename(ligand_filename)[:10] + '.pdb')  # '_pocket10.pdb')  # PDBId_Chain_rec.pdb
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol_litlmol(cls, ligand_rdmol, protein_filename='targets/sarm1_rec.pdb', **kwargs):
        return cls(protein_filename, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir=f'/homefs/home/{gt.getuser()}/scratch/tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        try:
            self.tmp_dir = os.path.realpath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        except FileNotFoundError:
            tmp_dir = f"/home/{gt.getuser()}/scratch/tmp/"
            self.tmp_dir = os.path.realpath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer
        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)
        # vina_dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        # vina_dock.pocket_center, vina_dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        # score, pose = vina_dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        score, pose = safe_dock(ligand_pdbqt, protein_pdbqt, self.center, self.size_x, self.size_y, self.size_z, mode=mode, exhaustiveness=exhaustiveness, **kwargs)
        return [{'affinity': score, 'pose': pose}]
