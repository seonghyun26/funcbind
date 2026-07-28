
import os
from collections import OrderedDict
import math
import torch

from funcbind.models.decoder import Decoder, _normalize_coords, _unnormalize_coords, get_atom_coords_batched, get_code_spatial
from funcbind.models.encoder import Encoder, MPEncoder, sample_posterior
from funcbind.models.unet3d import MPConv
from funcbind.utils.constants import is_antibody_dataset
from funcbind.utils.utils_sampling import filter_valid_mol, save_sdf_pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from typing import List


def create_nf_encoder(config, fabric):
    """
    Creates and initializes an encoder based on the provided configuration.

    Args:
        config (dict): A dictionary containing configuration parameters for the encoder and decoder.
            Expected keys include:
                - "dset": A dictionary with dataset-specific parameters:
                    - "code_dim" (int): The dimensionality of the bottleneck code.
                    - "n_channels" (int): The number of input channels.
                    - "grid_dim" (int): The dimensionality of the grid.
                - "encoder": A dictionary with encoder-specific parameters:
                    - "level_channels" (list): A list of channel sizes for each level.
                    - "spatial" (bool): Whether to use spatial encoding.
            - "debug" (bool): A flag indicating whether to run in debug mode.

        fabric (object): An object providing utility functions such as printing and model compilation.

    Returns:
        tuple: A tuple containing the initialized encoder and decoder models.
    """
    num_channels = get_num_channels(config)
    fabric.print(">> creating nf encoder", config["encoder"]["name"] if "name" in config["encoder"] else "vanilla_cnn")
    enc = Encoder(
        bottleneck_channel=config["decoder"]["code_dim"],
        out_channel=2 * config["decoder"]["code_dim"] if config["reg_weight"] != 0.0 else config["decoder"]["code_dim"],
        in_channels=num_channels,
        level_channels=config["encoder"]["level_channels"],
        downsample_map=config["encoder"].get("downsample_map", [False, False, False]),
    )
    n_params_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    fabric.print(f">> enc has {(n_params_enc/1e6):.02f}M parameters")

    return enc


def create_nf_decoder(config, fabric):
    decoder_type = config["decoder"].get("type", "inr")
    fabric.print(f">> creating nf decoder ({decoder_type})...")
    num_channels = get_num_channels(config)
    if "latent_grid_dim" in config["dset"]:
        n = 2 * sum(config["encoder"].get("downsample_map", [False, False, False]))
        latent_grid_dim = config["dset"]["latent_grid_dim"] // n if n > 0 else config["dset"]["latent_grid_dim"]
    else:
        latent_grid_dim = 16
    if decoder_type == "inr":
        dec = Decoder(
            n_channels=num_channels,
            grid_dim=config["dset"]["grid_dim"],
            hidden_dim=config["decoder"]["hidden_dim"],
            code_dim=config["decoder"]["code_dim"],
            coord_dim=config["decoder"]["coord_dim"],
            n_layers=config["decoder"]["n_layers"],
            input_scale=config["decoder"]["input_scale"],
            fabric=fabric,
            resolution=config["dset"]["resolution"],
            per_patch_coord=config["dset"].get("per_patch_coord", False),
            latent_grid_dim=latent_grid_dim,
        )
    elif decoder_type == "gaussian_splat":
        from funcbind.models.gaussian_splat_decoder import (
            ChannelWiseGaussianSplatDecoder3D,
        )

        dec = ChannelWiseGaussianSplatDecoder3D(
            n_channels=num_channels,
            code_dim=config["decoder"]["code_dim"],
            grid_dim=config["dset"]["grid_dim"],
            latent_grid_dim=latent_grid_dim,
            hidden_dim=config["decoder"].get("hidden_dim", 512),
            gaussians_per_voxel=config["decoder"].get(
                "gaussians_per_voxel", 1
            ),
            scale_min=config["decoder"].get("scale_min", 0.35),
            scale_max=config["decoder"].get("scale_max", 1.25),
            offset_bound=config["decoder"].get("offset_bound", 0.5),
            opacity_threshold=config["decoder"].get(
                "opacity_threshold", 0.01
            ),
            initial_opacity=config["decoder"].get(
                "initial_opacity", 0.05
            ),
            cutoff_sigma=config["decoder"].get("cutoff_sigma", 3.0),
            query_chunk_size=config["decoder"].get(
                "query_chunk_size", 512
            ),
            gaussian_chunk_size=config["decoder"].get(
                "gaussian_chunk_size", 512
            ),
            fabric=fabric,
            resolution=config["dset"]["resolution"],
        )
    else:
        raise ValueError(
            f"Unknown decoder type {decoder_type!r}; expected 'inr' or "
            "'gaussian_splat'"
        )
    n_params_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    fabric.print(f">> dec has {(n_params_dec/1e6):.02f}M parameters")
    return dec


def get_num_channels(config):
    return config["dset"]["n_channels"]


def create_receptor_encoder(config, n_channels_receptor):
    level_channels = config["denoiser"]["level_channels"]
    if config["denoiser"].get("res_factor_receptor", 1) >= 2:
        downsample_map = [True, False]
    else:
        downsample_map = config["denoiser"].get("downsample_map", [False, False])
    n_downsampling = sum(downsample_map)
    receptor_encoder = torch.nn.Sequential(
        MPConv(n_channels_receptor, level_channels[0], kernel=[3,3,3]),
        MPEncoder(
            level_channels[0],
            config["decoder"]["code_dim"],
            level_channels=level_channels[1:],
            resample_mode="down",
            n_downsampling=n_downsampling,
            downsample_first_block=downsample_map[0],
        )
    )
    return receptor_encoder


def create_neural_field(config, fabric):
    """
    Create and compile a neural field encoder and decoder based on the provided configuration and fabric.

    Args:
        config (dict): Configuration dictionary containing parameters for creating the encoder and decoder.
        fabric (object): An object that provides utility functions such as printing messages.

    Returns:
        tuple: A tuple containing the compiled encoder and decoder.
    """
    enc = create_nf_encoder(config, fabric)
    if "name" not in config["encoder"] or ("name" in config["encoder"] and config["encoder"]["name"] == "vanilla_cnn"):
        enc = torch.compile(enc)
        fabric.print(">> encoder compiled")

    dec = create_nf_decoder(config, fabric)
    if config["decoder"].get("compile", True):
        dec = torch.compile(dec)
        fabric.print(">> decoder compiled")
    else:
        fabric.print(">> decoder compilation disabled")

    return enc, dec


def load_neural_field(nf_checkpoint, fabric, config = None, input = None, setup_fabric=True):
    """
    Load and initialize the neural field encoder and decoder from a checkpoint.

    Args:
        nf_checkpoint (dict): The checkpoint containing the saved state of the neural field model.
        fabric (object): The fabric object used for setting up the modules.
        config (dict, optional): Configuration dictionary for initializing the encoder and decoder.
                                 If None, the configuration from the checkpoint will be used.

    Returns:
        tuple: A tuple containing the initialized encoder and decoder modules.
    """
    if config is None:
        config = nf_checkpoint["config"]
        if "subsampling_ratio" in config["dset"]:
            config["dset"]["latent_grid_dim"] = config["dset"]["grid_dim"] // config["dset"]["subsampling_ratio"]
            config["dset"]["latent_resolution"] = config["dset"]["resolution"] * config["dset"]["subsampling_ratio"]
    dec = create_nf_decoder(config, fabric)
    try:
        dec = load_network(nf_checkpoint, dec, fabric, net_name=f"dec_{input}" if input is not None else "dec")
    except KeyError as e:
        fabric.print(f">> Loading error dec: {e}.")

    if config["decoder"].get("compile", True):
        dec = torch.compile(dec)
    dec.eval()

    enc = create_nf_encoder(config, fabric)
    try:
        enc = load_network(nf_checkpoint, enc, fabric, net_name=f"enc_{input}" if input is not None else "enc")
    except KeyError as e:
        fabric.print(f">> Loading error enc: {e}.")
    enc = torch.compile(enc)
    enc.eval()

    if setup_fabric:
        dec = fabric.setup_module(dec)
        enc = fabric.setup_module(enc)

    return enc, dec


def infer_codes_batch(batch, enc, field_maker, config, xs = None, code_stats = None, to_cpu=False, save_log_var=True, deterministic=False):
    voxels = field_maker.compute_voxel_grid(batch["ligand"], num_channels=len(config["dset"]["elements"]))
    codes = enc(voxels)
    reg_weight = "reg_weight" in config and config["reg_weight"] != 0.0

    posteriors = None
    decoder_type = config["decoder"].get("type", "inr")
    requires_full_latent_grid = decoder_type == "gaussian_splat"
    if xs is not None and not requires_full_latent_grid:
        codes = get_code_spatial(xs, codes)
    if reg_weight:
        codes, posteriors = sample_posterior(codes, save_log_var=save_log_var, deterministic=deterministic)
    if code_stats is not None:
        codes = normalize_code(codes, code_stats)
    if to_cpu:
        codes = codes.cpu()

    return codes, posteriors


def infer_occupancies_batch(batch, field_maker, config, to_cpu=False):
    occs = _compute_occupancies_batched(batch["ligand"], field_maker, config["dset"]["n_channels"])
    if to_cpu:
        occs = occs.cpu()
    return occs


def _compute_occupancies_batched(mol_dict, field_maker, num_channels):
    # If the molecule is small enough, compute the occupancies directly
    # Otherwise, split the xs into smaller chunks and compute the occupancies separatey
    # This is done to avoid running out of memory
    if mol_dict["xs"].shape[1] < 100_000:
        return field_maker.compute_occupancies(mol_dict, num_channels=num_channels)
    else:
        # Split the molecule into smaller chunks with smaller xs
        n_points = mol_dict["xs"].shape[1]
        n_chunks = math.ceil(n_points / 100_000)
        chunk_size = math.ceil(n_points / n_chunks)
        occs = []
        mol_dict_ = {}
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_points)
            for key in mol_dict.keys():
                if key != "xs":
                    mol_dict_[key] = mol_dict[key]
                else:
                    mol_dict_[key] = mol_dict[key][:, start:end, :]
            occs.append(field_maker.compute_occupancies(mol_dict_, num_channels=num_channels))
        return torch.cat(occs, dim=1)


def get_points(batch, repeat_nb=1):
    # used only to train/val/eval neural fields
    xs = batch["ligand"]["xs"]
    if repeat_nb > 1:
        xs = xs.repeat(repeat_nb, 1, 1)
    return xs


def occs_to_mol(config, fabric, occs_point=None, vox_point=None, mols=[], seq=None, resolution=0.25):
    if occs_point is not None and vox_point is None:
        occs_point = occs_point.permute(0, 2, 1).reshape(-1, occs_point.size(2), config["dset"]["grid_dim"], config["dset"]["grid_dim"], config["dset"]["grid_dim"])
    else:
        occs_point = vox_point.clone()

    mol_inits = get_atom_coords_batched(
        occs_point, fabric=fabric, rad=config["dset"]["ligand_radius"] if "ligand_radius" in config["dset"] else 0.5,
        resolution=resolution)

    for b, mol_init in enumerate(mol_inits):
        if seq is not None:
            mol_init["seq"] = seq[b]
        if mol_init is not None:
            mol_init = _normalize_coords(mol_init, config["dset"]["grid_dim"])
            mol_init = _unnormalize_coords(mol_init, config["dset"]["grid_dim"], config["dset"]["resolution"])
            mols.append(mol_init)


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


def filter_mol_to_sdf_pdb(config, fabric, mols, center_coords=None, fname="samples", dirname=None, verbose=True):
    if dirname is None:
        dirname = config["dirname"]

    # Mol to SDF
    rdkmols = []
    list_smiles = []
    is_antibody = is_antibody_dataset(config)
    rdkmols = filter_valid_mol(mols, center_coords=center_coords, remove_fragment=False, valid_mols=rdkmols, valid_seq=list_smiles, fabric=fabric, is_ab=is_antibody, verbose=verbose)[0]
    seq_pred = save_sdf_pdb(rdkmols, dirname, fabric, fname=fname, to_pdb=is_antibody, verbose=verbose)
    return seq_pred


def load_network(
    checkpoint,
    net,
    fabric,
    net_name="dec",
):
    """
    Load a neural network's state dictionary from a checkpoint and update the network's parameters.

    Args:
        checkpoint (dict): A dictionary containing the checkpoint data.
        net (nn.Module): The neural network model to load the state dictionary into.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The key name for the network's state dictionary in the checkpoint. Defaults to "dec".
        is_compile (bool, optional): A flag indicating whether the network is compiled. Defaults to True.
        sd (str, optional): A specific key for the state dictionary in the checkpoint. If None, defaults to using net_name.

    Returns:
        nn.Module: The neural network model with the loaded state dictionary.
    """
    net_dict = net.state_dict()
    weight_first_layer_before = next(iter(net_dict.values())).sum()
    new_state_dict = OrderedDict()
    key = f"{net_name}_state_dict"
    for k, v in checkpoint[key].items():
        new_state_dict[k] = v

    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    weight_first_layer_after = next(iter(net_dict.values())).sum()
    assert weight_first_layer_before != weight_first_layer_after, "loading did not work"
    fabric.print(f">> loaded {net_name}")

    return net


def save_checkpoint(
    epoch,
    config,
    loss_tot,
    loss_min_tot,
    enc,
    dec,
    optim_enc,
    optim_dec,
    fabric,
    acc_iter=0,
    enc_ema=None,
    dec_ema=None,
):
    """
    Saves a model checkpoint if the current total loss is less than the minimum total loss.

    Args:
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training parameters.
        loss_tot (float): The current total loss.
        loss_min_tot (float): The minimum total loss encountered so far.
        enc (nn.Module): The encoder model.
        dec (nn.Module): The decoder model.
        optim_enc (torch.optim.Optimizer): The optimizer for the encoder.
        optim_dec (torch.optim.Optimizer): The optimizer for the decoder.
        fabric (object): An object responsible for saving the model state.

    Returns:
        float: The updated minimum total loss.
    """

    if loss_min_tot is None or loss_tot < loss_min_tot:
        if loss_min_tot is not None:
            loss_min_tot = loss_tot
        try:
            state = {
                "epoch": epoch,
                "dec_state_dict": dec.state_dict(),
                "dec_state_dict_ema": dec_ema.state_dict() if dec_ema is not None else None,
                "enc_state_dict": enc.state_dict(),
                "enc_state_dict_ema": enc_ema.state_dict() if enc_ema is not None else None,
                "optim_dec": optim_dec.state_dict(),
                "optim_enc": optim_enc.state_dict(),
                "config": config,
                "acc_iter": acc_iter,
            }
            fabric.save(os.path.join(config["dirname"], "model.pt"), state)
            fabric.print(">> saved checkpoint")
        except Exception as e:
            fabric.print(f"Error saving checkpoint: {e}")
    return loss_min_tot


def update_config_nf(config_nf, config):
    config_nf["debug"] = config["debug"]
    config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]
    config_nf["dset"]["input_dataset"] = config["dset"]["input_dataset"]
    config_nf["dset"]["use_single_dataset"] = config["dset"].get("use_single_dataset", None)
    config_nf["dset"]["cdrs"] = config["dset"].get("cdrs", ["H3"])
    config_nf["dset"]["cdrs_aug"] = config["dset"].get("cdrs_aug", [])
    config_nf["dset"]["data_aug"] = config["dset"].get("data_aug", True)
    return config_nf


def plot_xs(batch_tensor: np.ndarray,
            batch_index: int,
            output_dir: str = "output_plots",
            formats: List[str] = ['png', 'html']):
    """
    Creates a 3D scatter plot from a batch of coordinates and saves it to a file.
    This function can save both a static PNG image and an interactive HTML file.

    To display the interactive plot in a Jupyter Notebook cell, you can use:
    from IPython.display import HTML
    HTML(filename=f"{output_dir}/batch_{batch_index}.html")

    Args:
        batch_tensor (np.ndarray): A NumPy array of shape (N, 3) containing x, y, z coordinates.
        batch_index (int): The index of the current batch for titling and filenames.
        output_dir (str): The directory where the plot images will be saved.
        formats (List[str]): A list of file formats to save.
                             Supported formats: 'png', 'html'.
                             Defaults to ['png', 'html'].
    """
    # --- Input Validation ---
    if batch_tensor.ndim != 2 or batch_tensor.shape[1] != 3:
        raise ValueError(f"Input tensor must have shape (N, 3), but got {batch_tensor.shape}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Static PNG Plot (matplotlib) ---
    if 'png' in formats:
        # Extract coordinate data
        x_coords, y_coords, z_coords = batch_tensor[:, 0], batch_tensor[:, 1], batch_tensor[:, 2]

        # Create figure
        fig_mpl = plt.figure(figsize=(10, 8))
        ax = fig_mpl.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(x_coords, y_coords, z_coords, c='cyan', marker='o', s=5, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Customize plot
        ax.set_title(f'3D Scatter Plot for Batch {batch_index}', fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Z Coordinate', fontsize=12)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.grid(True)

        # Save the plot
        filename_png = os.path.join(output_dir, f"batch_{batch_index}.png")
        plt.tight_layout()
        plt.savefig(filename_png)
        plt.close(fig_mpl) # Free memory
        print(f"Static plot saved to '{filename_png}'")

    # --- Interactive HTML Plot (plotly) ---
    if 'html' in formats:
        # Create a pandas DataFrame for Plotly Express
        df = pd.DataFrame(batch_tensor, columns=['x', 'y', 'z'])

        # Create the 3D scatter plot
        fig_plotly = px.scatter_3d(df, x='x', y='y', z='z',
                                   title=f'Interactive 3D Scatter Plot for Batch {batch_index}',
                                   labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'z': 'Z Coordinate'},
                                   range_x=[-1, 1], range_y=[-1, 1], range_z=[-1, 1])

        # Customize marker appearance
        fig_plotly.update_traces(marker=dict(size=2,
                                             color='cyan',
                                             opacity=0.7,
                                             line=dict(color='black', width=1)))

        # Save the interactive plot
        filename_html = os.path.join(output_dir, f"batch_{batch_index}.html")
        fig_plotly.write_html(filename_html)
        print(f"Interactive plot saved to '{filename_html}'")