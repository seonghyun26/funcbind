import os
from typing import Tuple, Optional
import lightning as L
from omegaconf import OmegaConf
from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.strategies import DDPStrategy
import getpass as gt
import torch
import warnings
import omegaconf
import contextlib
from funcbind.utils.constants import ELEMENT_2_HASH_FUNCBIND, ELEMENT_2_VDW_RADIUS, PADDING_INDEX

# Suppress noisy Biopython PDB construction warnings globally (duplicate atoms/residues, etc.)
try:  # Keep optional to avoid hard dependency during import
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
except Exception:
    pass


def setup_fabric(config: dict, find_unused_parameters=False) -> L.Fabric:
    """
    Sets up the L.Fabric environment for distributed training.

    This function initializes the L.Fabric environment based on the number of available CUDA devices.
    If more than one CUDA device is available, it sets up a distributed data parallel (DDP) strategy
    with 8 devices. Otherwise, it uses a single device with an automatic strategy. The function also
    seeds the environment and prints the configuration and device information.

    Args:
        config (dict): Configuration dictionary containing the seed value.

    Returns:
        L.Fabric: The initialized L.Fabric environment.
    """
    logger = None
    if config["wandb"]:
       logger = WandbLogger(
           project="funcbind",
           entity=gt.getuser(),
           config=OmegaConf.to_container(config),
           name=config["exp_name"],
           dir=config["dirname"],
       )

    n_devs = config.get("n_devs") if "n_devs" in config else torch.cuda.device_count()
    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    strat_ = "ddp" if n_devs > 1 else "auto"
    if strat_ == "ddp" and find_unused_parameters:
        strat_ = DDPStrategy(find_unused_parameters=True)
    if n_devs >= 1:
        fabric = L.Fabric(
            devices=n_devs, num_nodes=n_nodes, strategy=strat_, accelerator="gpu", loggers=[logger], precision="bf16-mixed"
        )
    else:
        fabric = L.Fabric(accelerator="cpu", loggers=[logger], precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(config["seed"])
    if config["wandb"]:
        fabric.log("start", True)  # dummy command to launch logging in wandb
    fabric.print(f"config:\n{OmegaConf.to_yaml(config)}")
    fabric.print(f"world_size: {fabric.world_size} = n_nodes: {n_nodes} x n_devs: {n_devs}")

    return fabric


def recenter_structures(ligand: dict, receptor: Optional[dict], center_coords: torch.Tensor) -> Tuple[dict, dict]:
    """
    Recenter the ligand and target structures based on the provided center coordinates.

    Args:
        ligand (dict): Dictionary containing the ligand structure information.
        target (dict): Dictionary containing the target structure information.
        center_coords (torch.Tensor): Tensor representing the center coordinates.

    Returns:
        tuple: A tuple containing the centered ligand and centered target structures.
    """
    # subtract center of mass from ligand
    if ligand is not None:
        coords = ligand["coords"]
        center_coords_tiled = center_coords.unsqueeze(0).tile((coords.shape[0], 1))
        centered_ligand = {k: v for k, v in ligand.items()}
        centered_ligand["coords"] = coords - center_coords_tiled
    else:
        centered_ligand = None

    if receptor is None:
        return centered_ligand, None
    else:
        # subtract center of mass from target
        coords = receptor["coords"]
        center_coords_tiled = center_coords.unsqueeze(0).tile((coords.shape[0], 1))
        centered_receptor = {k: v for k, v in receptor.items()}
        centered_receptor["coords"] = coords - center_coords_tiled
        return centered_ligand, centered_receptor


def rotate_coords(ligand: dict, receptor: Optional[dict], rot_matrix:torch.Tensor=None) -> Tuple[dict, dict]:
    """Randomly rotate coordinates

    Args:
        ligand (dict): Dictionary containing ligand information.
            It should have a key "coords" with a torch.Tensor of shape [Nx3].
        receptor (dict): Dictionary containing receptor information.
            It should have a key "coords" with a torch.Tensor of shape [Nx3].

    Returns:
        tuple: A tuple containing the updated ligand and receptor dictionaries.
            The "coords" key in both dictionaries will be updated with the rotated coordinates.
    """
    if rot_matrix is None:
        from funcbind.utils.utils_sampling import random_rot_matrix
        rot_matrix = random_rot_matrix()

    coords_ligand = ligand["coords"]
    coords_ligand = torch.reshape(coords_ligand, (-1, 3))
    ligand["coords"] = torch.einsum("ij, kj -> ki", rot_matrix, coords_ligand)

    if receptor is None:
        return ligand, None

    coords_receptor = receptor["coords"]
    coords_receptor = torch.reshape(coords_receptor, (-1, 3))
    receptor["coords"] = torch.einsum("ij, kj -> ki", rot_matrix, coords_receptor)

    return ligand, receptor


def rotate_coords_single(ligand: dict, rot_matrix:torch.Tensor=None) -> Tuple[dict, dict]:
    """Randomly rotate coordinates

    Args:
        ligand (dict): Dictionary containing ligand information.
            It should have a key "coords" with a torch.Tensor of shape [Nx3].

    Returns:
        tuple: A tuple containing the updated ligand and receptor dictionaries.
            The "coords" key in both dictionaries will be updated with the rotated coordinates.
    """
    assert rot_matrix is not None, "rot_matrix should be provided"

    coords_ligand = ligand["coords"]
    coords_ligand = torch.reshape(coords_ligand, (-1, 3))
    ligand["coords"] = torch.einsum("ij, kj -> ki", rot_matrix, coords_ligand)

    return ligand


def translate_coords(ligand: dict, receptor: Optional[dict], delta: float = 1.0) -> Tuple[dict, dict]:
    """
    Translates the coordinates of the ligand and receptor by adding random noise.

    Args:
        ligand (dict): Dictionary containing the ligand coordinates.
        receptor (dict): Dictionary containing the receptor coordinates.
        delta (float, optional): Maximum magnitude of the random noise. Defaults to 1.

    Returns:
        tuple: Tuple containing the updated ligand and receptor dictionaries.
    """
    noise = (torch.rand((1, 3)) - 1 / 2) * 2 * delta

    ligand["coords"] += noise.repeat(ligand["coords"].shape[0], 1)

    if receptor is None:
        return ligand, None

    receptor["coords"] += noise.repeat(receptor["coords"].shape[0], 1)

    return ligand, receptor


def atomChannelsToRadius(atoms_channel: torch.Tensor, hashing: dict = ELEMENT_2_HASH_FUNCBIND) -> torch.Tensor:
    """
    Convert atom channels to corresponding atomic radii.

    Args:
        atoms_channel (torch.Tensor): Tensor containing atom channels.
        hashing (dict): Dictionary mapping atom indices to element symbols.

    Returns:
        torch.Tensor: Tensor containing atomic radii corresponding to the atom channels.
    """
    radius = []
    element_ids = [k for k in hashing.keys()]
    for atom_channel in atoms_channel:
        if atom_channel < len(element_ids):
            element = element_ids[atom_channel]
            radius.append(ELEMENT_2_VDW_RADIUS[element])
        else:
            radius.append(PADDING_INDEX)
    return torch.Tensor(radius)

def is_in_contact(molecule1, molecule2, interface_threshold: float=5.0):
    # Calculate pairwise distances using cdist
    pairwise_dists = torch.cdist(molecule1["coords"], molecule2["coords"])

    # Check if any residue is in contact
    in_contact = (pairwise_dists < interface_threshold).any()

    return in_contact

def filter_atoms_by_distance_mask(
    molecule: dict,
    max_dim: float = 25,
) -> Tuple[dict, dict]:
    mask = torch.logical_or(molecule["coords"] < -max_dim, molecule["coords"] > max_dim)
    mask = mask.any(dim=1)
    # remove residues in molecule that are outside of the box
    for key in molecule.keys():
        if key in ["coords", "atoms_channel", "radius", "atom_residue_ids"]:
            molecule[key] = molecule[key][~mask]
    return molecule


def out_of_box(
    molecule: dict,
    max_dim: float = 25,
) -> Tuple[dict, dict]:
    mask = torch.logical_or(molecule["coords"] < -max_dim, molecule["coords"] > max_dim)
    mask = mask.any(dim=1)
    return mask.sum().item() > 0


def makedir(path: str) -> None:
    """Create a directory if it doesn't exist.

    Args:
        path (str): The path to the directory.
    """
    os.makedirs(path, exist_ok=True)


def supress_stdout(func):
    """
    A decorator that suppresses the standard output (stdout) of a function.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


def overwrite_config(cfg_ckpt, cfg):
    """
    Recursively update cfg_ckpt with cfg.
    """
    for key, value in cfg.items():
        if isinstance(value, omegaconf.dictconfig.DictConfig) and key in cfg_ckpt:
            # If the key exists in cfg_ckpt and both are dictionaries, recurse
            overwrite_config(cfg_ckpt[key], value)
        else:
            # Otherwise, overwrite or add the key in cfg_ckpt
            cfg_ckpt[key] = value
    return cfg_ckpt



def log_epoch(
    config: dict,
    epoch: int,
    loss_train: float,
    miou_train: float,
    loss_val: float,
    miou_val: float,
    elapsed_time: float,
    fabric: object,
    loss_rec_train: float,
    loss_reg_train: float,
    loss_rec_val: float,
    loss_reg_val: float,
    acc_iter: int,
    optimizer_enc,
    optimizer_dec
    ) -> None:
    """
    Logs the training and validation metrics for a given epoch.

    Args:
        config (dict): Configuration dictionary containing dataset and experiment details.
        epoch (int): The current epoch number.
        loss_train (float): Training loss for the current epoch.
        miou_train (float): Training mean Intersection over Union (mIoU) for the current epoch.
        loss_val (float): Validation loss for the current epoch.
        miou_val (float): Validation mean Intersection over Union (mIoU) for the current epoch.
        elapsed_time (float): Time elapsed since the start of training in seconds.
        fabric (object): An object for logging metrics, such as a Weights and Biases (wandb) logger.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log = f"| {config['exp_name']} [{epoch}/{config['n_epochs']}]" +\
        f" | train_loss: {loss_train:0.3e} | train_miou: {miou_train:0.3e}"
    if loss_rec_train is not None and loss_reg_train is not None:
        log += f" | train_rec_loss: {loss_rec_train:0.3e} | train_reg_loss: {loss_reg_train:0.3e}"
    if loss_val is not None:
        log += f" | val_loss: {loss_val:0.3e} | val_miou: {miou_val:0.3e}"
    if loss_rec_val is not None and loss_reg_val is not None:
        log += f" | val_rec_loss: {loss_rec_val:0.3e} | val_reg_loss: {loss_reg_val:0.3e}"
    log += f" | {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s"

    if config["wandb"]:
        fabric.log_dict({
            "trainer/global_step": epoch,
            "train_loss": loss_train,
            "train_miou": miou_train,
            "train_rec_loss": loss_rec_train,
            "train_reg_loss": loss_reg_train,
            "val_loss": loss_val,
            "val_miou": miou_val,
            "val_rec_loss": loss_rec_val,
            "val_reg_loss": loss_reg_val,
            "acc_iter": acc_iter,
            "acc_iter_normalized": acc_iter / config["dset"]["batch_size"],
            "lr_enc": optimizer_enc.param_groups[0]["lr"],
            "lr_dec": optimizer_dec.param_groups[0]["lr"],
        })
    fabric.print(log)
