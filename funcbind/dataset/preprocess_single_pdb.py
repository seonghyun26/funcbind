import os
import argparse
from biopandas.pdb import PandasPdb
from funcbind.dataset.preprocess_sabdab import CDR, process_file
from pyuul import utils
import torch
from openbabel import pybel
from funcbind.utils.utils_base import recenter_structures, rotate_coords, atomChannelsToRadius, translate_coords
from funcbind.dataset.preprocess_mcp_pair import get_conformation
from funcbind.utils.constants import ELEMENT_2_HASH_FUNCBIND, PEPTIDE_ALPHABET
from funcbind.utils.utils_base import filter_atoms_by_distance_mask
from funcbind.utils.utils_sampling import save_pocket_and_ligand
import tempfile


def preprocess_single_pdb(source_pdb_path: str, data_dir: str, dname: str = "single_pdbs", h_chain: str = "H", l_chain: str = "L") -> str:
    """
    Preprocesses a single PDB file by splitting it into antibody (Ab) and antigen (Ag) chains,
    renumbering atoms and residues, and saving the processed files.

    Args:
        source_pdb_path (str): The path to the source PDB file.
        data_dir (str): The data directory path.
        dname (str): The name of the sub-directory to save the processed files.
        h_chain (str): The chain identifier for the heavy chain of the antibody.
        l_chain (str): The chain identifier for the light chain of the antibody.

    Returns:
        str: Full path to the processed tensor file (.pt) created for this PDB.
    """
    print(f"Processing {source_pdb_path}")

    # Load the PDB file
    if 's3://' in source_pdb_path:
        os.system(f"aws s3 cp {source_pdb_path} {data_dir}/{dname}/")
        source_pdb_path = os.path.join(data_dir, dname, os.path.basename(source_pdb_path))
        ppdb = PandasPdb().read_pdb(source_pdb_path)
        if "temp" in dname:
            # Remove the downloaded file only in temp folder
            os.remove(source_pdb_path)
    else:
        ppdb = PandasPdb().read_pdb(source_pdb_path)

    # Split into ab and ag
    ab = ppdb.df["ATOM"][ppdb.df["ATOM"]["chain_id"].isin([h_chain, l_chain])].copy()
    ag = ppdb.df["ATOM"][~ppdb.df["ATOM"]["chain_id"].isin([h_chain, l_chain])].copy()

    orig_ab_chains = ab["chain_id"].unique()
    orig_ag_chains = ag["chain_id"].unique()

    # Renumber atoms to be contiguous for each chain
    ab["atom_number"] = range(1, len(ab) + 1)
    ag["atom_number"] = range(1, len(ag) + 1)

    # Rename Ab chains to be 'H' and 'L'
    ab["chain_id"] = ["H" if chain == h_chain else "L" for chain in ab["chain_id"]]

    # Rename Ag chains to be 'A', 'B', 'C', etc.
    chain_map = {chain: chr(65 + i) for i, chain in enumerate(ag["chain_id"].unique())}
    ag["chain_id"] = ag["chain_id"].map(chain_map)

    # Ensure each chain residue_number begins from 1
    for chain in ab["chain_id"].unique():
        ab.loc[ab["chain_id"] == chain, "residue_number"] += 1 - ab.loc[ab["chain_id"] == chain, "residue_number"].values[0]
    for chain in ag["chain_id"].unique():
        ag.loc[ag["chain_id"] == chain, "residue_number"] += 1 - ag.loc[ag["chain_id"] == chain, "residue_number"].values[0]

    # Create output dirs
    pdb_folder = os.path.join(data_dir, dname)
    out_folder = os.path.join(data_dir, dname)
    file_name = os.path.basename(source_pdb_path).replace(".pdb", "")

    ab_path = os.path.join(pdb_folder, file_name + f"_{''.join(orig_ab_chains)}.pdb")
    ag_path = os.path.join(pdb_folder, file_name + f"_{''.join(orig_ag_chains)}.pdb")
    out_path = os.path.join(out_folder, file_name + f"_{''.join(orig_ab_chains)}_{''.join(orig_ag_chains)}.pt")

    print(f"Writing Ab: `{ab_path}`, Ag: `{ag_path}`, Processed: `{out_path}`")
    ppdb.df["ATOM"] = ab
    ppdb.to_pdb(ab_path, records=['ATOM'], gz=False, append_newline=True)
    ppdb.df["ATOM"] = ag
    ppdb.to_pdb(ag_path, records=['ATOM'], gz=False, append_newline=True)
    process_file([ab_path, ag_path, out_path])

    return os.path.abspath(out_path)


def read_sm_structures(pdb_file, sdf_file):
    """
    Read structures from PDB and SDF files.

    Args:
        pdb_file (str): Path to the PDB file.
        sdf_file (str): Path to the SDF file.

    Returns:
        tuple: A tuple containing the ligand and target structures.
            - ligand (dict): Dictionary containing information about the ligand structure.
            - target (dict): Dictionary containing information about the target structure.
    """
    # ligand
    ligand = None
    if sdf_file is not None:
        coords, atname = utils.parseSDF(sdf_file)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND)
        mask = atoms_channel[0] < 7
        ligand = {
            "id": sdf_file,
            "coords": coords[0][mask],
            "atoms_channel": atoms_channel[0][mask].type(torch.uint8),
            "radius": .5 * torch.ones_like(atoms_channel[0][mask])
        }

    # target
    coords, atname = utils.parsePDB(pdb_file)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND).type(torch.uint8)
    mask = atoms_channel[0] < 4  # pocket only has C, O, N, S
    target = {
        "id": pdb_file,
        "coords": coords[0][mask].clone(),
        "atoms_channel": atoms_channel[0][mask].type(torch.uint8),
        "radius": atomChannelsToRadius(atoms_channel[0][mask], ELEMENT_2_HASH_FUNCBIND),
    }

    return ligand, target


def read_ab_structures(pdb_file: str, h_chain: str = "H", l_chain: str = "L", grid_dim: int = 128, resolution: float = 0.25, data_dir: str = None, dname: str = None, cdr = CDR.H3) -> tuple:
    """
    Process antibody PDB file using preprocess_single_pdb internally.

    Args:
        pdb_file: Path to the PDB file containing antibody-antigen complex
        h_chain: Heavy chain identifier
        l_chain: Light chain identifier

    Returns:
        tuple: (ligand_data, target_data) where ligand_data is antibody and target_data is antigen
    """
    # Create temporary directory for processing
    remove_temp_dir = False
    if data_dir is None:
        data_dir = tempfile.mkdtemp()
        remove_temp_dir = True
    if dname is None:
        dname = "temp_processing"

    try:
        # Use preprocess_single_pdb to do the heavy lifting
        out_path = preprocess_single_pdb(
            source_pdb_path=pdb_file,
            data_dir=data_dir,
            dname=dname,
            h_chain=h_chain,
            l_chain=l_chain,
        )

        # Load the processed data
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Expected processed tensor at {out_path}, but it does not exist.")
        processed_data = torch.load(out_path)

        # Extract antibody (ligand) and antigen (target) from processed data
        ag, ab = processed_data

        # more processing here
        cdr_region_mask = (ab["cdr_region"] == cdr)
        cdr_mask = cdr_region_mask[ab["atom_residue_ids"]]
        assert (cdr_mask.sum() != 0), "No CDR H3 found"
        recentered_ab, recentered_ag, recentered_cdr_region_mask = reduce_size_ab((grid_dim * resolution) // 2, ab, ag, cdr_mask, cdr_region_mask)
        recentered_ab["cdr_h3_seq"] = "".join(PEPTIDE_ALPHABET[idx] for idx in recentered_ab["sequences"][cdr_region_mask])

        recentered_ag = {
            "coords": torch.cat([recentered_ag["coords"], recentered_ab["coords"][~recentered_cdr_region_mask]], dim=0),
            "atoms_channel": torch.cat([recentered_ag["atoms_channel"], recentered_ab["atoms_channel"][~recentered_cdr_region_mask]], dim=0),
        }
        recentered_ab = {
            "coords": recentered_ab["coords"][recentered_cdr_region_mask],
            "atoms_channel": recentered_ab["atoms_channel"][recentered_cdr_region_mask],
            "cdr_h3_seq": recentered_ab["cdr_h3_seq"],
        }
        return recentered_ab, recentered_ag

    finally:
        # Clean up temporary directory
        if remove_temp_dir:
            import shutil
            shutil.rmtree(data_dir)


def reduce_size_ab(max_dim, ab, ag, cdr_mask, cdr_region_mask):
    # Code to reduce size of the antibody and antigen to the sampled CDR region
    initial_coords = ab["coords"][cdr_mask].mean(0)
    recentered_ab, recentered_ag = recenter_structures(ab, ag, initial_coords)
    recentered_ab = filter_atoms_by_distance_mask(recentered_ab, max_dim=max_dim + 5)
    recentered_cdr_region_mask = cdr_region_mask[recentered_ab["atom_residue_ids"]]
    if ag is not None:
        recentered_ag = filter_atoms_by_distance_mask(recentered_ag, max_dim=max_dim + 5)
    recentered_ab, recentered_ag = recenter_structures(recentered_ab, recentered_ag, -initial_coords)  # in order to stitch back the sampled CDR and framework at sampling time
    return recentered_ab, recentered_ag, recentered_cdr_region_mask


def preprocess_ligand_receptor(
    target_pdb_file: str,
    ligand_sdf_file: str = None,
    receptor_center = None,
    aug: bool = False,
    out_dir: str = None,
    # Antibody-specific parameters
    h_chain: str = "H",
    l_chain: str = "L",
    modality: str = "ab",
) -> tuple:
    """
    Unified preprocessing function that handles both small molecule-protein and antibody-antigen complexes.

    Args:
        target_pdb_file (str): Path to the PDB file
        ligand_sdf_file (str, optional): Path to SDF file. If None, antibody mode is used.
        receptor_center (list, optional): [x, y, z] coordinates for pocket center
        aug (bool, optional): Whether to apply augmentation. Defaults to False.
        out_dir (str, optional): Output directory to save files. Defaults to None.
        h_chain (str, optional): Heavy chain identifier for antibody mode. Defaults to "H".
        l_chain (str, optional): Light chain identifier for antibody mode. Defaults to "L".
        is_antibody (bool, optional): Force antibody mode. Auto-detected if None.

    Returns:
        tuple: (ligand, receptor) - preprocessed data ready for sampling
    """
    # Determine processing mode
    if "ab" in modality:  # by default infers only CDR H3
        print(f"Processing in ANTIBODY mode: {target_pdb_file}")
        ligand, target = read_ab_structures(target_pdb_file, h_chain, l_chain, data_dir="dataset/data/", dname="test_raw_pdbs")
        data_type = 2   # if "neurips" in modality else 2 # TODO: for now it is the same for neurips and ab
    elif modality == "mcp":
        print(f"Processing in MCP mode: {target_pdb_file} + {ligand_sdf_file}")
        prot = next(pybel.readfile("pdb", target_pdb_file))
        target = get_conformation(prot, mode="torch")
        if ligand_sdf_file is not None:
            try:
                mcp = next(pybel.readfile("sdf", ligand_sdf_file))
                ligand = get_conformation(mcp, mode="torch")
            except Exception as e:
                ligand = None
                assert receptor_center is not None, "Receptor center is needed when ligand .sdf is not available"
        else:
            print("Ligand .sdf is not available")
            ligand = None
            assert receptor_center is not None, "Receptor center is needed when ligand .sdf is available"
        data_type = 1
    else:
        print(f"Processing in SMALL MOLECULE mode: {target_pdb_file} + {ligand_sdf_file}")
        ligand, target = read_sm_structures(target_pdb_file, ligand_sdf_file)
        data_type = 0

    # center reference of frame to center of mass of ligand
    center_coords = ligand["coords"].mean(axis=0) if receptor_center is None else torch.Tensor(receptor_center)
    ligand, target = recenter_structures(ligand, target, center_coords)
    if aug:
        ligand, target = rotate_coords(ligand, target)
        ligand, target = translate_coords(ligand, target, 1.0)

    if "ab" in modality:
        ligand_id = target_pdb_file.split("/")[-1].split(".")[0][:-len("_A")]
        target_id = target_pdb_file.split("/")[-1].split(".")[0][:-len("_HL_A")] + "_A"
    else:
        ligand_id = os.path.basename(target_pdb_file)
        target_id = os.path.basename(target_pdb_file)

    # box molecules
    if ligand is not None:
        ligand = filter_atoms_by_distance_mask(ligand)
        ligand = {
            "coords": ligand["coords"],
            "atoms_channel": ligand["atoms_channel"],
            "data_type": data_type,
            "id": ligand_id,
            "cdr_h3_seq": ligand["cdr_h3_seq"] if "cdr_h3_seq" in ligand else "",
        }
        ligand_radius = 1.0
        if ligand_radius > 0:
            ligand['radius'] = ligand_radius * torch.ones_like(ligand["atoms_channel"] ).float()
        else:
            ligand['radius'] = atomChannelsToRadius(ligand["atoms_channel"])
    if target is not None:
        target = filter_atoms_by_distance_mask(target)
        target = {
            "coords": target["coords"],
            "atoms_channel": target["atoms_channel"],
            "data_type": data_type,
            "id": target_id,
            "center_coords": center_coords,
        }
        receptor_radius = 1.0
        if receptor_radius > 0:
            target['radius'] = receptor_radius * torch.ones_like(target["atoms_channel"]).float()
        else:
            target['radius'] = atomChannelsToRadius(target["atoms_channel"])

    # save pdb and sdf on out_dir
    if "ab" in modality:
        if "s3://" not in target_pdb_file:
            folder = os.path.dirname(target_pdb_file)
            for file in [target_pdb_file, folder + "/" + ligand["id"] + ".pdb", folder + "/" + target["id"] + ".pdb"]:
                print(f"Saving {file} to {out_dir}")
                save_pocket_and_ligand(file, None, out_dir)
    else:
        print(f"Saving {target_pdb_file} and {ligand_sdf_file} to {out_dir}")
        save_pocket_and_ligand(target_pdb_file, ligand_sdf_file, out_dir)

    print(f"Center coords: {center_coords}")

    return ligand, target
