from copy import deepcopy
from functools import partial
from funcbind.utils.utils_base import rotate_coords
import torch
from torch import nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from funcbind.models.mfn import GaborNet
from lightning.fabric import Fabric
from funcbind.utils.utils_vis import visualize_ligand_receptor


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 2048,
        code_dim: int = 2048,
        coord_dim: int = 3,
        n_layers: int = 6,
        input_scale: int = 64,
        grid_dim: int = 32,
        reshape_out: int = 1,
        fabric=None,
        resolution: float = 0.25,
        per_patch_coord: bool = False,
        latent_grid_dim: int = 16,
    ):
        """
        Initialize the INRDecoder module.

        Args:
            n_channels (int): Number of input channels.
            hidden_dim (int): Dimension of the hidden layers.
            code_dim (int): Dimension of the code layers.
            coord_dim (int): Dimension of the coordinate input.
            n_layers (int, optional): Number of layers in the network. Defaults to 1.
            inr_type (str, optional): Type of the INR network. Can be "fnet" or "gnet". Defaults to "fnet".
            input_scale (int, optional): Scale factor for the input coordinates. Defaults to 64.
            per_patch_coord (bool, optional): If True, coordinates are transformed to [-1, 1] within each patch. Defaults to False.
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.code_dim = code_dim
        self.grid_dim = grid_dim
        self.fabric = fabric
        self.reshape_out = reshape_out
        self.resolution = resolution
        self.per_patch_coord = per_patch_coord
        self.latent_grid_dim = latent_grid_dim
        self.coords = get_grid(self.grid_dim)[1].unsqueeze(0).to(self.fabric.device)
        fabric.print(">> coords shape:", self.coords.shape)
        self.net = GaborNet(
            self.coord_dim,
            self.hidden_dim,
            self.code_dim,
            n_channels,
            n_layers,
            input_scale=input_scale,
        )

    def _transform_coords_to_per_patch(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Transform coordinates from global [-1, 1] to per-patch [-1, 1].

        Args:
            xs (torch.Tensor): Input coordinates in global [-1, 1] space.
            latent_grid_dim (int): Dimension of the latent grid (s).

        Returns:
            torch.Tensor: Transformed coordinates in per-patch [-1, 1] space.
            This ensures each patch uses the full [-1, 1] coordinate range internally.
        """
        # Apply transformation to all dimensions at once (vectorized)
        # Determine which patch each coordinate belongs to (consistent with get_code_spatial)
        patch_idx = torch.floor((xs + 1) * (self.latent_grid_dim - 1) / 2).long()
        # Clamp to valid range [0, latent_grid_dim-1]
        patch_idx = torch.clamp(patch_idx, 0, self.latent_grid_dim - 1)

        # Calculate the start of each patch in global coordinates
        patch_start = 2 * patch_idx.float() / (self.latent_grid_dim - 1) - 1

        # Calculate patch width
        patch_width = 2.0 / (self.latent_grid_dim - 1)

        # Transform to [0, 1] within patch, then to [-1, 1]
        pos_in_patch = (xs - patch_start) / patch_width
        xs_transformed = pos_in_patch * 2 - 1

        return xs_transformed

    def forward(self, x, codes=None) -> torch.Tensor:
        """
        Forward pass of the INRDecoder module.

        Args:
            x: Input tensor.
            codes: Optional tensor of codes.
            subset: Optional subset of the output.

        Returns:
            Tensor: Output tensor.
        """
        # Transform coordinates to per-patch [-1, 1] if requested
        if self.per_patch_coord:
            x = self._transform_coords_to_per_patch(x)

        out = self.net(x, codes)
        if self.reshape_out > 1:
            out = out.reshape(-1, out.size(1), self.reshape_out)
        return out

    def render_code(self, codes: torch.Tensor, batch_size_render, fabric=None, verbose=True) -> torch.Tensor:
        """
        Create a voxel grid from the (normalized) codes

        Args:
            codes: Input codes.

        Returns:
            Tensor: Rendered codes.
        """
        # PS: code need to be unnormalized before rendering
        with torch.no_grad():
            if verbose:
                fabric.print(f">> Rendering molecules - batches of {batch_size_render}")
            if codes.device != self.fabric.device:
                codes = codes.to(self.fabric.device)
            xs = self.coords.reshape(1, -1, 3)
            if codes.size(0) != xs.size(0):
                xs_spatial = xs.expand(codes.size(0), -1, -1)
            else:
                xs_spatial = xs

            codes = get_code_spatial(xs_spatial, codes)

            pred = self.forward_batched(xs, codes, batch_size_render=batch_size_render)
            grid = pred.permute(0, 2, 1).reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
        return grid

    def forward_batched(self, xs, codes, batch_size_render=100_000, threshold=None, to_cpu=True):
        """
        When memory is limited, render the grid in batches.
        """
        pred_list = []
        batched_xs = torch.split(xs, batch_size_render, dim=1)
        batched_codes = torch.split(codes, batch_size_render, dim=1)
        for x, code in zip(batched_xs, batched_codes):
            pred_batched = self(x, code)
            if to_cpu:
                pred_batched = pred_batched.cpu()
            if threshold is not None:
                pred_batched[pred_batched < threshold] = 0
            pred_list.append(pred_batched)
        pred = torch.cat(pred_list, dim=1)
        return pred


    def codes_to_molecules_batched(self, codes, unnormalize, config, fabric=None, rand_rots=None, batch_size_render_codes=None, verbose=True, target_dirname=None):
        if batch_size_render_codes is None:
            batch_size_render_codes = codes.size(0)
        batched_codes = torch.split(codes, batch_size_render_codes, dim=0)
        mols = []
        for batched_code in tqdm(batched_codes):
            mols += self.codes_to_molecules(
                batched_code, unnormalize=unnormalize, fabric=fabric, config=config, rand_rots=rand_rots, verbose=verbose, target_dirname=target_dirname
            )
        return mols


    def codes_to_molecules(self, codes, unnormalize, config, fabric=None, rand_rots=None, verbose=True, target_dirname=None, voxel_name=None):
        codes = codes.detach()
        if unnormalize:
            codes = self.unnormalize_code(codes).detach()

        mols_dict, codes_dict, indices_dict, rotations = self.codes_to_mols_dict(
            codes=codes, config=config, fabric=fabric, rand_rots=rand_rots, verbose=verbose, target_dirname=target_dirname, voxel_name=voxel_name
        )
        if verbose:
            fabric.print("(n_atoms, n_samples): ", end=" ")
            for key, value in sorted(mols_dict.items()):
                fabric.print(f"({key}, {len(value)})", end=" ")
            fabric.print()

        # Refine coordinates
        mols = self._refine_coords(
            grouped_mol_inits=mols_dict, grouped_codes=codes_dict, grouped_indices=indices_dict, maxiter=200, grid_dim=self.grid_dim,
            resolution=self.resolution, fabric=fabric, verbose=False
        )

        # unrotate gen_mols
        if len(rotations) > 0:
            mols = unrotate_gen_mols(mols, rotations, fabric)

        return mols


    def _process_mol_inits(self, mol_inits, codes, rand_rots, fabric, max_atoms=2000):
        """
        Process molecular initializations and organize them into dictionaries by atom count.

        Args:
            mol_inits: List of molecular initialization dictionaries
            codes: Tensor of codes corresponding to each molecule
            rand_rots: Random rotations (can be None)
            max_atoms: Maximum number of atoms allowed per molecule
            fabric: Fabric object for printing

        Returns:
            tuple: (mols_dict, codes_dict, indices_dict, rotations)
        """
        mols_dict = defaultdict(list)
        codes_dict = defaultdict(list)
        indices_dict = defaultdict(list)
        rotations = [] if rand_rots is None or (rand_rots is not None and len(rand_rots) > 1) else rand_rots

        for idx, mol_init in enumerate(mol_inits):
            if mol_init is not None:
                mol_init = _normalize_coords(mol_init, self.grid_dim)
                num_coords = int(mol_init["coords"].size(1))
                if num_coords <= max_atoms:
                    mols_dict[num_coords].append(mol_init)
                    codes_dict[num_coords].append(codes[idx].cpu())
                    indices_dict[num_coords].append(idx)  # Track index to avoid reshuffling when refining coords
                    if rand_rots is not None and len(rand_rots) > 1:
                        rotations.append(rand_rots[idx])
                else:
                    fabric.print(f"Molecule {idx} has more than {max_atoms} atoms")
            else:
                fabric.print(f"No atoms found in grid {idx}")

        return mols_dict, codes_dict, indices_dict, rotations


    def codes_to_mols_dict(
        self,
        codes: torch.Tensor,
        config=None,
        fabric=None,
        rand_rots=None,
        verbose=True,
        target_dirname=None,
        voxel_name="pred",
    ) -> tuple:
        """
        Converts a batch of codes to grids and extracts atom coordinates.

        Args:
            batched_codes (torch.Tensor): The batch of codes to convert to grids.
            mols_dict (dict, optional): A dictionary to store molecule objects. Defaults to defaultdict(list).
            codes_dict (dict, optional): A dictionary to store the corresponding codes. Defaults to defaultdict(list).

        Returns:
            tuple: A tuple containing the grid shape, molecule objects, dictionaries of molecule objects, codes, and indices.
        """
        # 1. render grid
        grids = self.render_code(codes, config["sampling"]["batch_size_render"], fabric, verbose=verbose)
        if config["sampling"].get("save_voxel", False):
            # Lazy import to avoid circular dependency
            from funcbind.train_nf import save_grids
            fabric.print(f">> Saving voxels to {target_dirname}")
            for i in range(grids.shape[0]):
                save_grids(grids[i].unsqueeze(0) if voxel_name == "gt" else None, grids[i].unsqueeze(0) if voxel_name == "pred" else None, target_dirname, i, config)

        # 2. find peaks (atom coordinates)
        mol_inits = get_atom_coords_batched(
            grids, fabric, rad=config["dset"]["ligand_radius"] if "ligand_radius" in config["dset"] else 0.5,
            resolution=self.resolution, flexible=config["sampling"].get("flexible", True)
        )
        mols_dict, codes_dict, indices_dict, rotations = self._process_mol_inits(
            mol_inits, codes, rand_rots, fabric
        )

        return mols_dict, codes_dict, indices_dict, rotations


    def _refine_coords(
        self,
        grouped_mol_inits: dict,
        grouped_codes: dict,
        grouped_indices: dict,
        maxiter: int = 10,
        grid_dim: int = 32,
        resolution: int = 0.25,
        fabric=None,
        verbose=True
    ) -> list:
        """
        Refines the coordinates of the molecules based on the initial coordinates and codes.

        Args:
            mol_inits (list): A list of initial molecule coordinates.
            codes (torch.Tensor): Tensor containing the codes for each molecule.
            maxiter (int, optional): Maximum number of iterations for refinement. Defaults to 10.

        Returns:
            list: A list of refined molecules.
        """
        if verbose:
            fabric.print(">> Refining molecules")
        refined = {}
        for key, mols in grouped_mol_inits.items():
            try:
                coords = self._refine_coords_batch(grouped_mol_inits[key], grouped_codes[key], maxiter=maxiter, fabric=fabric)
                for i in range(coords.size(0)):
                    mols[i]["coords"] = coords[i].unsqueeze(0)
                    mols[i] = _unnormalize_coords(mols[i], grid_dim, resolution)
                    refined[grouped_indices[key][i]] = mols[i]
            except Exception as e:
                fabric.print(f"Error refinement: {e}")
                for i, mol in enumerate(mols):
                    try:
                        mol = _unnormalize_coords(mol, grid_dim, resolution)
                        refined[grouped_indices[key][i]] = mol
                    except Exception:
                        fabric.print(f"Error unnormalization for {i}/{len(mols)}")
        # Reorder based on original indices
        ordered_refined = [refined[idx] for idx in sorted(refined)]
        return ordered_refined


    def _refine_coords_batch(
        self,
        mols_init: list,
        codes: list,
        maxiter: int = 10,
        batch_size_refinement: int = 100,
        fabric=None,
    ) -> torch.Tensor:
        """
        Refines the coordinates of all atoms in a molecule at once.

        Args:
            mol_init (dict): The initial molecule representation containing the atoms channel and coordinates.
            code (torch.Tensor): The code used for refining the coordinates.
            maxiter (int, optional): The maximum number of iterations for refining the coordinates. Defaults to 10.

        Returns:
            dict: The refined molecule representation with updated coordinates.
        """
        num_batches = len(mols_init) // batch_size_refinement
        if len(mols_init) % batch_size_refinement != 0:
            nb_iter = num_batches + 1
        else:
            nb_iter = num_batches
        refined_coords = []

        for i in range(nb_iter):
            min_bound = i * batch_size_refinement
            max_bound = (len(mols_init) if i == num_batches else (i + 1) * batch_size_refinement)
            coords = torch.stack(
                [mols_init[j]["coords"].squeeze(0) for j in range(min_bound, max_bound)], dim=0,
            ).to(fabric.device)
            coords_init = coords.clone()
            coords.requires_grad = True

            with torch.no_grad():
                atoms_channel = torch.stack(
                    [mols_init[j]["atoms_channel"] for j in range(min_bound, max_bound)], dim=0,
                ).to(fabric.device)
                occupancy = (
                    torch.nn.functional.one_hot(atoms_channel.long(), self.n_channels).float().squeeze()
                ).to(fabric.device)
                code = torch.stack(
                    [codes[j] for j in range(min_bound, max_bound)], dim=0
                ).to(fabric.device)
                code = get_code_spatial(coords, code)

            def closure():
                optimizer.zero_grad()
                pred = self(coords, code)
                loss = -(pred * occupancy).max(dim=2).values.mean()
                loss.backward()
                return loss

            optim_factory = partial(
                torch.optim.LBFGS,
                history_size=10,
                max_iter=4,
                line_search_fn="strong_wolfe",
                lr=0.01,
            )
            optimizer = fabric.setup_optimizers(optim_factory([coords]))
            tol, loss = 1e-4, 1e10
            for iter in range(maxiter):
                prev_loss = loss
                loss = optimizer.step(closure)
                if abs(loss - prev_loss).item() < tol:
                    # print(f"Refine coords converges at iter {iter} loss {loss.item()}")
                    break
                if (coords - coords_init).abs().max() > 1:
                    print("Refine coords diverges, so use initial coordinates...")
                    coords = coords_init
                    break
            # print(f"Refine coords iter {iter} loss {loss.item()}")
            refined_coords.append(coords.detach().cpu())
        return torch.cat(refined_coords, dim=0)

    def set_code_stats(self, code_stats: dict) -> None:
        """
        Set the code statistics.

        Args:
            code_stats: Code statistics.
        """
        self.code_stats = code_stats

    def unnormalize_code(self, codes: torch.Tensor):
        """
        Unnormalizes the given codes based on the provided normalization parameters.

        Args:
            codes (torch.Tensor): The codes to be unnormalized.
            code_stats (dict): The statistics for the codes.

        Returns:
            torch.Tensor: The unnormalized codes.
        """
        mean, std = (
            self.code_stats["mean"].to(codes.device),
            self.code_stats["std"].to(codes.device),
        )
        return codes * std + mean


########################################################################################

## auxiliary functions

def unrotate_gen_mols(gen_mols, rotations, fabric):
    unrotate_gen_mols = []
    for i, mol in enumerate(gen_mols):
        gen_mol_ = deepcopy(mol)
        gen_mol_["coords"] = gen_mol_["coords"].to(fabric.device)
        gen_mol_, _ = rotate_coords(gen_mol_, receptor=None, rot_matrix=rotations[i].T.to(fabric.device) if len(rotations) > 1 else rotations[0].T.to(fabric.device))
        gen_mol_["coords"] = gen_mol_["coords"].unsqueeze(0).cpu()
        unrotate_gen_mols.append(gen_mol_)
    return unrotate_gen_mols


def get_atom_coords_batched(
    grids: torch.Tensor,
    fabric: Fabric,
    resolution: float = 0.25,
    rad: float = 0.5,
    verbose: bool = True,
    flexible: bool = True,
) -> list[dict]:
    """ Batched peakfinding on grids, with shape [mols, atom_channels, g, g, g].
        Returns a list of structure dicts.
    """
    threshold = 0.8 * rad if rad > 0.25 else 0.05
    order = 2 if not flexible else 1
    peaks = find_peaks_batched(grids, fabric, resolution=resolution, threshold=threshold, flexible=flexible, order=order)
    # (mols, atom channels, g, g, g)
    peaks_ = [peaks_to_structure_dict(peak, rad=rad) for peak in peaks]
    if verbose:
        try:
            fabric.print("total number of peaks", sum(peak["coords"].shape[1] for peak in peaks_))
        except Exception as e:
            fabric.print(f"Error printing total number of peaks: {e}")
    return peaks_


def peaks_to_structure_dict(peaks: torch.Tensor, rad=0.5) -> dict:
    """ peaks: [atom_channels, g, g, g].
        Returns a structure dict.

        current version only works for fixed radius (ie, all atoms with same radius rad)
    """
    coords = []
    atoms_channel = []
    radius = []

    for channel_idx in range(peaks.shape[0]):
        px, py, pz = torch.where(peaks[channel_idx] > 0)
        if px.numel() > 0:
            px, py, pz = px.float(), py.float(), pz.float()
            coords.append(torch.stack([px, py, pz], dim=1))
            atoms_channel.append(
                torch.full((px.shape[0],), channel_idx, dtype=torch.float32)
            )
            radius.append(torch.full((px.shape[0],), rad, dtype=torch.float32))

    if not coords:
        return None

    structure = {
        "coords": torch.cat(coords, dim=0).unsqueeze(0),
        "atoms_channel": torch.cat(atoms_channel, dim=0).unsqueeze(0),
        "radius": torch.cat(radius, dim=0).unsqueeze(0),
    }

    return structure


def suppress_nearby_peaks(peaks: torch.Tensor, intensities: torch.Tensor, min_distance: float = 2.0):
    """Simple, fast non-maximum suppression to remove nearby peaks."""
    if not torch.any(peaks):
        return peaks

    # Get peak coordinates as a long tensor for indexing
    peak_coords_long = torch.nonzero(peaks, as_tuple=False)

    # Create a float version for distance calculations
    peak_coords_float = peak_coords_long.float()

    # Use the long tensor for indexing intensities
    peak_intensities = intensities[peak_coords_long[:, 0], peak_coords_long[:, 1], peak_coords_long[:, 2]]

    sorted_indices = torch.argsort(peak_intensities, descending=True)

    keep = torch.ones(len(peak_coords_long), dtype=torch.bool, device=peaks.device)
    min_dist_sq = min_distance * min_distance

    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]

        if not keep[idx_i]:
            continue

        # Use the float tensor for distance calculations
        sq_distances = torch.sum((peak_coords_float[sorted_indices[i + 1:]] - peak_coords_float[idx_i]) ** 2, dim=1)

        close_indices = sorted_indices[i + 1:][sq_distances <= min_dist_sq]
        keep[close_indices] = False

    output = torch.zeros_like(peaks)

    # Use the long tensor to create the final output
    kept_coords = peak_coords_long[keep]

    if kept_coords.numel() > 0:
        output[kept_coords[:, 0], kept_coords[:, 1], kept_coords[:, 2]] = True

    return output


def find_peaks_batched(
    grids: torch.Tensor,
    fabric: Fabric,
    order: int = 1,
    threshold = 0.7,
    resolution: float = 0.25,
    visualize: bool = False,
    flexible: bool = True,
) -> torch.Tensor:
    """ Finds peaks for grids: (mols, atom channels, g, g, g) """
    grids[grids < threshold] = 0
    grids = grids.to(fabric.device)

    with fabric.device:
        maxpool = torch.nn.MaxPool3d(2 * order + 1, stride=1, padding=order)
        output = maxpool(grids)
    if flexible:
        peaks = ((torch.abs(grids - output) <= 0.1) & (grids > 0))

        # Remove peaks that are too close to each other
        min_distance = 1.0 / resolution   # 1.0A minimum distance between peaks
        for batch_idx in range(peaks.shape[0]):
            for channel_idx in range(peaks.shape[1]):
                if peaks[batch_idx, channel_idx].sum() > 0:
                    peaks[batch_idx, channel_idx] = suppress_nearby_peaks(
                        peaks[batch_idx, channel_idx], grids[batch_idx, channel_idx], min_distance
                    )
    else:
        peaks = ((torch.abs(grids - output) == 0.0) & (grids > 0))

    # Visualize output and peaks using visualize_ligand_receptor
    if visualize:
        try:
            # Visualize detected peaks
            peaks_float = peaks[0].float().cpu()  # Convert boolean to float
            visualize_ligand_receptor(
                ligand=grids[0].cpu(),
                receptor=peaks_float,
                name="peak_debug_overlay",
                dirname="figures",
                threshold=threshold,  # Use same threshold for grids
                to_png=True,
                to_html=True,
                sparse=True,
                peaks=True
            )
            fabric.print("Saved: figures/peak_debug_overlay.html")

        except Exception as e:
            fabric.print(f"Visualization error: {e}")

    return peaks


def _normalize_coords(mol, grid_dim):
    mol["coords"] -= (grid_dim - 1) / 2
    mol["coords"] /= grid_dim / 2
    return mol


def _unnormalize_coords(mol, grid_dim, resolution=0.25):
    mol["coords"] *= grid_dim / 2
    mol["coords"] *= resolution
    return mol


def get_code_spatial(
    xs: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    Get the code tensor from the encoder for the given input coordinates.

    Args:
        xs (torch.Tensor): The input coordinates tensor.
        codes (torch.Tensor): The code tensor.
        config (dict): Configuration dictionary containing model parameters.
        enc (nn.Module): The encoder model.
        fabric (object): An object with a print method for logging.

    Returns:
        torch.Tensor: The code tensor for the given input coordinates.
    """
    xs_idx = torch.floor((xs + 1) * (codes.size(-1) - 1) / 2).to(torch.int)
    batch_indices = torch.arange(codes.size(0)).unsqueeze(1).expand(xs.size(0), xs.size(1))
    x_coords = xs_idx[..., 0]
    y_coords = xs_idx[..., 1]
    z_coords = xs_idx[..., 2]
    codes = codes[batch_indices, :, x_coords, y_coords, z_coords]
    if not codes.is_contiguous():
        codes = codes.contiguous()
    return codes


def get_grid(grid_dim):
    discrete_grid = (np.arange(grid_dim) - (grid_dim // 2)) / (grid_dim // 2)
    full_grid = torch.Tensor(
        [[a, b, c] for a in discrete_grid for b in discrete_grid for c in discrete_grid]
    )
    return discrete_grid, full_grid
