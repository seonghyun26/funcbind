from funcbind.utils.constants import CHANNEL_TO_ATM_NB
import numpy as np
from funcbind.utils import reconstruct
from rdkit import Chem
import torch
from openbabel import pybel


def mol2rdkit_obabel(
    mol: dict,
    remove_fragment: bool = False,
    remove_atoms_too_close: bool = False,
    return_ob_mol: bool = False
) -> Chem.rdchem.Mol:
    """
    Convert a molecule dictionary to an RDKit molecule using Open Babel.
    This function uses Open Babel to convert the molecule dictionary to an RDKit molecule.
    It reconstructs the molecule using the provided coordinates and atom types.
    If the reconstruction fails, it returns None.

    Args:
        mol (dict): The molecule dictionary containing coordinates, atom types, and other information.
        sdf_file (str, optional): The path to the SDF file. Defaults to None.

    Returns:
        rdkmol (RDKit molecule): The RDKit molecule representation of the input molecule.

    Raises:
        reconstruct.MolReconsError: If molecule reconstruction fails.
    """
    if remove_atoms_too_close:
        mol = remove_atoms_too_close_mol_dict(mol)

    pred_pos = [[m[0].item(), m[1].item(), m[2].item()] for m in mol["coords"][0]]
    pred_atom_type = [CHANNEL_TO_ATM_NB[m.item()] for m in mol["atoms_channel"][0].int()]

    rdkmol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type)

    if remove_fragment:
        smiles = Chem.MolToSmiles(rdkmol)
        if "." in smiles:
            return None

    if return_ob_mol:
        # Convert RDKit → Open Babel via MOL block
        mol_block = Chem.MolToMolBlock(rdkmol)
        ob_mol = pybel.readstring("mol", mol_block)
        return rdkmol, ob_mol
    else:
        return rdkmol, None



def remove_atoms_too_close_rdkit(mol: Chem.rdchem.Mol, max_distance: float = 0.4) -> Chem.rdchem.Mol:
    """
    Removes atoms that are too close to each other in a molecule using RDKit.

    Args:
        mol (Chem.rdchem.Mol): The input molecule.

    Returns:
        Chem.rdchem.Mol: The modified molecule with close atoms removed.
    """
    try:
        dists = Chem.rdmolops.Get3DDistanceMatrix(mol)
        row, col = np.where((dists > 0) & (dists < max_distance)) # Removed eye condition as Get3DDistanceMatrix has 0 on diagonal

        if len(row) == 0:
            return mol

        # Identify atoms to remove (atoms with too many close neighbors)
        atoms_to_remove = set()
        for i, j in zip(row, col):
            # Choose one of the atoms to remove (e.g., the one with higher index)
            atoms_to_remove.add(max(i, j))

        # Sort in descending order to avoid index shifting issues
        atoms_to_remove = sorted(atoms_to_remove, reverse=True)

        # Remove atoms
        print(f"Removing {len(atoms_to_remove)} atoms")
        edit_mol = Chem.EditableMol(mol)
        for idx in atoms_to_remove:
            edit_mol.RemoveAtom(int(idx))

        return edit_mol.GetMol()
    except Exception as e:
        print(f"Error in remove_atoms_too_close_rdkit: {e}")
        return mol


def remove_atoms_too_close_mol_dict(mol: dict, max_distance: float = 0.4) -> dict:
    """
    Removes atoms that are too close directly from mol dict.

    Args:
        mol (dict): Molecule dictionary with "coords" and "atoms_channel" keys
        max_distance (float): Maximum distance threshold

    Returns:
        dict: Updated mol dict with close atoms removed
    """
    coords = mol["coords"][0]  # Shape: [N, 3]
    atoms_channel = mol["atoms_channel"][0]  # Shape: [N]

    if len(coords) == 0:
        return mol

    try:
        # Calculate pairwise distances
        n_atoms = len(coords)
        coords_np = coords.cpu().numpy()

        dists = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords_np[i] - coords_np[j])
                dists[i, j] = dist
                dists[j, i] = dist

        # Find atoms that are too close
        row, col = np.where((dists > 0) & (dists < max_distance))

        if len(row) == 0:
            return mol

        # Identify atoms to remove
        atoms_to_remove = set()
        for i, j in zip(row, col):
            atoms_to_remove.add(max(i, j))

        # Sort in descending order to avoid index shifting issues
        atoms_to_remove = sorted(atoms_to_remove, reverse=True)

        print(f"Removing {len(atoms_to_remove)} atoms from mol dict")

        # Create mask for atoms to keep
        keep_mask = torch.ones(n_atoms, dtype=torch.bool)
        for idx in atoms_to_remove:
            keep_mask[idx] = False

        # Filter coordinates and atoms
        new_mol = mol.copy()
        new_mol["coords"] = [coords[keep_mask]]
        new_mol["atoms_channel"] = [atoms_channel[keep_mask]]

        return new_mol

    except Exception as e:
        print(f"Error in remove_atoms_too_close_mol_dict: {e}")
        return mol
