"""One-target FuncBind proof of concept with real holo-density conditioning.

The original pretrained FuncBind denoiser, receptor encoder, ligand encoder,
and INR decoder are frozen. A small adapter is fitted on CrossDocked test
target 69 (5MGL/7MU), mapping a frozen VoxBind ChannelViT representation of
the aligned holo density into a residual on FuncBind's receptor condition.

This is deliberately a diagnostic, not a generalization result: the 2Fo-Fc
holo map contains the reference ligand and the adapter is overfit to the same
target. The generated artifacts make both limitations explicit.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from funcbind.compare_decoders import (
    _atomic_sparse_occupancy_dump,
    atom_recovery_metrics,
    reconstruction_metrics,
)
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.distributions.distributions import UniformPlusNormal
from funcbind.models.denoiser import FuncBind, add_noise_to_code
from funcbind.models.decoder import get_atom_coords_batched
from funcbind.models.encoder import sample_posterior
from funcbind.train_fb import get_label
from funcbind.utils.constants import PADDING_INDEX
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_fb import (
    create_field_makers,
    load_unet,
    num_classes_funcbind,
)
from funcbind.utils.utils_nf import (
    create_nf_decoder,
    create_nf_encoder,
    filter_mol_to_sdf_pdb,
)
from funcbind.utils.utils_sampling import get_receptor_encoding, normalize_code


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIGAND_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P", "Br"]
_ARM_LABELS = {
    "original_funcbind": "Original FuncBind",
    "holo_density_funcbind": "Holo-density adapter",
    "apo_density_funcbind": "Apo-density adapter",
    "transfer_adapter_funcbind": "Holo-trained adapter on this density",
}


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(x) for x in value.shape]


def _load_models(config: DictConfig, fabric):
    """Load only inference weights; mmap avoids loading the 57 GiB optimizer."""
    fb_path = _resolve_path(config.fb_pretrained_path) / "checkpoint.pth.tar"
    nf_path = _resolve_path(config.nf_pretrained_path) / "model.pt"
    if not fb_path.is_file():
        raise FileNotFoundError(fb_path)
    if not nf_path.is_file():
        raise FileNotFoundError(nf_path)

    fabric.print(f">> memory-mapping FuncBind checkpoint: {fb_path}")
    fb_checkpoint = torch.load(
        fb_path, map_location="cpu", weights_only=False, mmap=True
    )
    model_config = OmegaConf.merge(
        OmegaConf.create(fb_checkpoint["config"]),
        OmegaConf.create(OmegaConf.to_container(config, resolve=False)),
    )
    model_config.dset.data_dir = str(_resolve_path(config.data_dir))
    model_config.dset.data_aug = False
    model_config.dset.rebalance = False
    model_config.dset.batch_size = 1
    model_config.dset.num_workers = 0
    model_config.dset.use_single_dataset = "xdocked"

    code_stats = {}
    for key, value in fb_checkpoint["code_stats"].items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        code_stats[key] = (
            value.to(fabric.device) if torch.is_tensor(value) else value
        )

    num_classes = num_classes_funcbind(model_config)
    model = FuncBind(
        model_config,
        code_stats=code_stats,
        fabric=fabric,
        num_classes=num_classes,
    )
    load_unet(fb_checkpoint, model, fabric, sd="state_dict_ema")
    del fb_checkpoint
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model = fabric.setup_module(model)

    fabric.print(f">> memory-mapping neural-field checkpoint: {nf_path}")
    nf_checkpoint = torch.load(
        nf_path, map_location="cpu", weights_only=False, mmap=True
    )
    nf_config = OmegaConf.create(nf_checkpoint["config"])
    nf_config.dset.data_dir = str(_resolve_path(config.data_dir))
    nf_config.dset.data_aug = False
    nf_config.dset.rebalance = False
    nf_config.dset.batch_size = 1
    nf_config.dset.num_workers = 0
    nf_config.dset.use_single_dataset = "xdocked"
    nf_config.decoder.type = "inr"
    nf_config.decoder.compile = False

    encoder = create_nf_encoder(nf_config, fabric)
    decoder = create_nf_decoder(nf_config, fabric)
    encoder.load_state_dict(nf_checkpoint["enc_state_dict"], strict=True)
    decoder.load_state_dict(nf_checkpoint["dec_state_dict"], strict=True)
    del nf_checkpoint
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    encoder = fabric.setup_module(encoder)
    decoder = fabric.setup_module(decoder)
    decoder_module = _unwrap(decoder)
    decoder_module.set_code_stats(code_stats)

    return (
        model,
        encoder,
        decoder_module,
        model_config,
        nf_config,
        num_classes,
        fb_path,
        nf_path,
    )


def _load_density_encoder(config: DictConfig, fabric):
    voxbind_root = _resolve_path(config.density.voxbind_python_root)
    if str(voxbind_root) not in sys.path:
        sys.path.insert(0, str(voxbind_root))
    from voxbind.models.density_vit import DensityViT

    checkpoint_path = _resolve_path(config.density.encoder_checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    encoder = DensityViT(
        grid_dim=64,
        patch_size=8,
        n_in_channels=13,
        c_out=16,
        dim=512,
        depth=12,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        pos_encoding="learnable",
        patch_embed_mode="channel_group",
        channel_groups=(7, 4, 1, 1),
        channel_group_dropout=0.0,
        n_memory_tokens=0,
    )
    raw = checkpoint["encoder_state_dict_ema"]
    stripped = {
        key[len("encoder.") :]: value
        for key, value in raw.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(stripped, strict=True)
    del checkpoint
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return fabric.setup_module(encoder), checkpoint_path


def _make_target_batch(nf_config: DictConfig, target_index: int, fabric):
    dataset = DatasetOmni(
        nf_config,
        split="test",
        sample_points=True,
        sample_full_grid=True,
        rebalance=False,
    )
    if not 0 <= target_index < len(dataset):
        raise IndexError(
            f"target_index={target_index} outside test set of size {len(dataset)}"
        )
    return fabric.to_device(collate_fn([dataset[target_index]])), len(dataset)


def _ligand_code(batch, encoder, field_maker, model) -> torch.Tensor:
    with torch.no_grad():
        voxels = field_maker.compute_voxel_grid(
            batch["ligand"], num_channels=len(LIGAND_ELEMENTS)
        )
        moments = encoder(voxels)
        code, _ = sample_posterior(
            moments,
            sample_posterior=False,
            save_log_var=False,
            deterministic=True,
        )
        code = normalize_code(code, _unwrap(model).code_stats)
    return code.detach()


def _ligand_occupancy_mask(
    batch: dict[str, Any],
    *,
    grid_dim: int,
    resolution: float,
    radius_scale: float,
    device,
) -> torch.Tensor:
    """FuncBind's ligand occupancy field evaluated on the density-crop grid.

    Same functional form as ``FieldMaker.compute_occupancies`` -- the union of
    per-atom Gaussians ``1 - prod(1 - exp(-(d / (0.93 r))^2))`` -- but pooled
    over elements so each voxel carries one occupancy weight in [0, 1].
    """
    channels = batch["ligand"]["atoms_channel"][0]
    valid = channels < len(LIGAND_ELEMENTS)
    coords = batch["ligand"]["coords"][0][valid].to(device).double()
    radius = batch["ligand"]["radius"][0][valid].to(device).double()
    radius = radius * float(radius_scale)

    points = _grid_points(grid_dim, resolution, device).double()
    exponent = (torch.cdist(points, coords) / (radius * 0.93)).square()
    exponent = exponent.clamp(max=10.0)
    log_term = torch.where(
        exponent < 10.0,
        torch.log1p(-torch.exp(-exponent)),
        torch.zeros_like(exponent),
    )
    occupancy = 1.0 - torch.exp(log_term.sum(1))
    return occupancy.reshape(grid_dim, grid_dim, grid_dim).float()


def _grid_points(grid_dim: int, resolution: float, device) -> torch.Tensor:
    axis = (
        torch.arange(grid_dim, device=device, dtype=torch.float32)
        - (grid_dim - 1) / 2.0
    ) * resolution
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def _min_distance_to_atoms(
    points: torch.Tensor, coords: torch.Tensor, chunk: int = 32768
) -> torch.Tensor:
    """Nearest-atom distance per point, chunked to bound peak memory."""
    if coords.numel() == 0:
        return torch.full((points.shape[0],), float("inf"), device=points.device)
    parts = [
        torch.cdist(points[start : start + chunk], coords).min(dim=1).values
        for start in range(0, points.shape[0], chunk)
    ]
    return torch.cat(parts)


def _apply_apo_mask(
    density: np.ndarray,
    batch: dict[str, Any],
    config: DictConfig,
    *,
    grid_dim: int,
    resolution: float,
    device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Erase ligand density from a holo map to emulate the apo state.

    The ligand occupancy field is used as a soft mask and the vacated voxels
    are refilled with the crop's own bulk-solvent level, so the result keeps
    the receptor density and the map's noise statistics but no longer shows
    the bound ligand.
    """
    occupancy = _ligand_occupancy_mask(
        batch,
        grid_dim=grid_dim,
        resolution=resolution,
        radius_scale=float(config.radius_scale),
        device=device,
    )
    mask = occupancy.clamp(0.0, 1.0)
    if float(config.mask_floor) > 0.0:
        # Harden the core so no ligand peak survives a soft blend.
        mask = torch.where(
            mask >= float(config.mask_floor), torch.ones_like(mask), mask
        )

    original = torch.from_numpy(density).to(device)
    points = _grid_points(grid_dim, resolution, device)
    channels = batch["ligand"]["atoms_channel"][0]
    ligand_coords = batch["ligand"]["coords"][0][
        channels < len(LIGAND_ELEMENTS)
    ].to(device)
    ligand_distance = _min_distance_to_atoms(points, ligand_coords).reshape(
        grid_dim, grid_dim, grid_dim
    )

    receptor_channels = batch["receptor"]["atoms_channel"][0]
    receptor_coords = batch["receptor"]["coords"][0][
        receptor_channels != PADDING_INDEX
    ].to(device)
    receptor_distance = _min_distance_to_atoms(points, receptor_coords).reshape(
        grid_dim, grid_dim, grid_dim
    )

    solvent_distance = float(config.solvent_min_distance)
    solvent = (ligand_distance > solvent_distance) & (
        receptor_distance > solvent_distance
    )
    if int(solvent.sum()) >= int(config.min_solvent_voxels):
        fill_value = float(original[solvent].median())
        fill_source = f"median of {int(solvent.sum())} bulk-solvent voxels"
    else:
        fill_value = float(original.median())
        fill_source = "median of the whole crop (too few bulk-solvent voxels)"

    masked = original * (1.0 - mask) + fill_value * mask

    envelope = ligand_distance <= float(config.envelope_radius)
    peak = float(config.leakage_sigma)
    atom_index = torch.round(
        ligand_coords / resolution + (grid_dim - 1) / 2.0
    ).long().clamp(0, grid_dim - 1)
    before = original[atom_index[:, 0], atom_index[:, 1], atom_index[:, 2]]
    after = masked[atom_index[:, 0], atom_index[:, 1], atom_index[:, 2]]

    diagnostics = {
        "enabled": True,
        "definition": (
            "FuncBind ligand occupancy field pooled over elements; masked "
            "voxels refilled with the crop's bulk-solvent level"
        ),
        "radius_scale": float(config.radius_scale),
        "mask_floor": float(config.mask_floor),
        "fill_value": fill_value,
        "fill_source": fill_source,
        "voxels_touched_fraction": float((mask >= 0.01).float().mean()),
        "voxels_fully_masked_fraction": float((mask >= 0.99).float().mean()),
        "ligand_atom_density_before": float(before.mean()),
        "ligand_atom_density_after": float(after.mean()),
        "ligand_atom_density_after_max": float(after.max()),
        "envelope_radius": float(config.envelope_radius),
        "leakage_sigma": peak,
        "envelope_above_sigma_before": float(
            (original[envelope] >= peak).float().mean()
        ),
        "envelope_above_sigma_after": float(
            (masked[envelope] >= peak).float().mean()
        ),
        "unmasked_voxels": int((mask <= 0.0).sum()),
        "unmasked_voxels_unchanged": bool(
            torch.equal(original[mask <= 0.0], masked[mask <= 0.0])
        ),
    }
    return (
        masked.cpu().numpy().astype(np.float32),
        mask.cpu().numpy().astype(np.float32),
        diagnostics,
    )


def _build_density_input(
    batch: dict[str, Any],
    density_np: np.ndarray,
    voxbind_root: Path,
    fabric,
) -> torch.Tensor:
    if str(voxbind_root) not in sys.path:
        sys.path.insert(0, str(voxbind_root))
    from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore
    from voxbind.voxelizer import Voxelizer

    density = torch.from_numpy(density_np)[None, None].to(fabric.device)
    gradmag = per_sample_zscore(gradient_magnitude3d(density))

    receptor = {
        "coords": batch["receptor"]["coords"],
        "atoms_channel": batch["receptor"]["atoms_channel"],
        "radius": batch["receptor"]["radius"].clone(),
    }
    channel = receptor["atoms_channel"].long()
    radius_lut = torch.tensor(
        [1.70, 1.52, 1.55, 1.80],
        device=channel.device,
        dtype=receptor["radius"].dtype,
    )
    valid = (channel >= 0) & (channel < 4)
    receptor["radius"][valid] = radius_lut[channel[valid]]

    voxelizer = Voxelizer(
        grid_dim=64,
        resolution=0.25,
        radius=0.5,
        cubes_around=8,
        device=str(fabric.device),
        backend="pyuul",
    )
    pocket_voxels = voxelizer(receptor, num_channels=4)
    ligand_unknown = torch.zeros(
        (1, 7, 64, 64, 64),
        device=fabric.device,
        dtype=pocket_voxels.dtype,
    )
    density_input = torch.cat(
        [
            ligand_unknown,
            pocket_voxels,
            density.to(pocket_voxels.dtype),
            gradmag.to(pocket_voxels.dtype),
        ],
        dim=1,
    )
    if density_input.shape != (1, 13, 64, 64, 64):
        raise RuntimeError(f"unexpected density input {_tensor_shape(density_input)}")
    return density_input


class DensityConditionAdapter(torch.nn.Module):
    """512x8^3 ChannelViT tokens -> residual 128x16^3 receptor condition."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(512, 192, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(192, 192, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(192, 128, kernel_size=1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        delta_small = self.net(feature)
        return F.pad(delta_small, (4, 4, 4, 4, 4, 4))


def _fit_adapter(
    adapter,
    optimizer,
    density_feature: torch.Tensor,
    receptor_base: torch.Tensor,
    ligand_target: torch.Tensor,
    label: torch.Tensor | None,
    model,
    config: DictConfig,
    fabric,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    steps = int(config.density.overfit_steps)
    report_every = int(config.density.report_every)
    reg = float(config.density.residual_regularization)
    history: list[dict[str, float]] = []
    adapter.train()
    model.eval()
    started = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sigma = _unwrap(model).sigma_distribution.sample((1,)).to(
            fabric.device
        )
        noisy = add_noise_to_code(ligand_target, sigma=sigma)
        with fabric.autocast():
            delta = adapter(density_feature)
            receptor_fused = receptor_base + delta
            prediction, logvar = model(
                ligand_encoding=noisy,
                sigma=sigma,
                receptor_encoding=receptor_fused,
                classes=label,
                return_logvar=True,
                cfg_dropout=False,
            )
            sigma5 = sigma.reshape(-1, 1, 1, 1, 1)
            weight = (sigma5.square() + 1.0) / sigma5.square()
            squared_error = (prediction - ligand_target).square()
            denoise_loss = (weight / logvar.exp() * squared_error + logvar).mean()
            residual_loss = delta.float().square().mean()
            loss = denoise_loss + reg * residual_loss
        fabric.backward(loss)
        fabric.clip_gradients(
            adapter,
            optimizer,
            max_norm=float(config.density.gradient_clip_norm),
        )
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == steps:
            item = {
                "step": step,
                "loss": float(loss.detach()),
                "denoise_loss": float(denoise_loss.detach()),
                "residual_rms": float(residual_loss.detach().sqrt()),
                "sigma": float(sigma.detach()),
                "elapsed_seconds": time.time() - started,
            }
            history.append(item)
            fabric.print(
                f">> adapter {step:04d}/{steps}: "
                f"loss={item['loss']:.5f}, "
                f"residual_rms={item['residual_rms']:.4f}"
            )

    adapter.eval()
    with torch.no_grad(), fabric.autocast():
        receptor_fused = receptor_base + adapter(density_feature)
    return receptor_fused.detach(), history


def _sample_codes(
    model,
    receptor_encoding: torch.Tensor,
    label: torch.Tensor | None,
    y_init: torch.Tensor,
    config: DictConfig,
    *,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
) -> torch.Tensor:
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    sampler = hydra.utils.instantiate(config.sampler)
    sampler.t_steps = sampler.t_steps.to(y_init.device)

    def score_fn(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        sigma_batch = torch.ones(
            batch_size, device=y.device, dtype=y.dtype
        ) * sigma
        xhat = model(
            ligand_encoding=y,
            sigma=sigma_batch,
            receptor_encoding=receptor_encoding,
            classes=label,
            cfg_dropout=False,
        )
        return (xhat - y) / sigma_batch.reshape(-1, 1, 1, 1, 1).square()

    with torch.no_grad():
        sampled = sampler.sample(score_fn, y_init=y_init.clone())["sample"]
    return sampled.float()


def _density_cloud(
    path: Path, density: np.ndarray, threshold: float, resolution: float = 0.25
) -> None:
    indices = np.argwhere(density >= threshold)
    coordinates = (indices.astype(np.float32) - 31.5) * resolution
    values = density[tuple(indices.T)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            coordinates=coordinates.astype(np.float32),
            values=values.astype(np.float16),
            threshold=np.asarray(threshold, dtype=np.float32),
            resolution=np.asarray(resolution, dtype=np.float32),
        )
    os.replace(temporary, path)


def _atoms_dump(path: Path, atoms: dict[str, torch.Tensor] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if atoms is None:
        coords = np.zeros((0, 3), dtype=np.float32)
        channels = np.zeros((0,), dtype=np.int16)
    else:
        coords = atoms["coords"][0].detach().float().cpu().numpy()
        channels = (
            atoms["atoms_channel"][0].detach().short().cpu().numpy()
        )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, coordinates=coords, channels=channels)
    os.replace(temporary, path)


def _evaluate_code(
    name: str,
    code: torch.Tensor,
    decoder,
    batch: dict[str, Any],
    target_flat: torch.Tensor,
    query_points: torch.Tensor,
    output_dir: Path,
    config: DictConfig,
    nf_config: DictConfig,
    fabric,
) -> dict[str, Any]:
    condition_dir = output_dir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    unnormalized = decoder.unnormalize_code(code.to(fabric.device))
    with torch.no_grad(), fabric.autocast():
        grid = decoder.render_code(
            unnormalized,
            batch_size_render=int(config.density.full_render_chunk_size),
            fabric=fabric,
            verbose=True,
        ).float()
    prediction_flat = grid.flatten(2).transpose(1, 2)
    atoms_grid = get_atom_coords_batched(
        grid.clone(),
        fabric,
        rad=float(nf_config.dset.ligand_radius),
        resolution=float(nf_config.dset.resolution),
        verbose=False,
        flexible=True,
    )[0]
    metrics = {
        "reconstruction": reconstruction_metrics(
            prediction_flat, target_flat
        ),
        "atom_recovery": atom_recovery_metrics(
            atoms_grid,
            batch["ligand"],
            grid_dim=int(nf_config.dset.grid_dim),
            resolution=float(nf_config.dset.resolution),
            match_distance=float(config.density.atom_match_distance),
        ),
    }
    _atomic_sparse_occupancy_dump(
        condition_dir / "occupancy_points.npz",
        query_points=query_points,
        occupancy=prediction_flat,
        threshold=float(config.density.occupancy_point_threshold),
        grid_dim=int(nf_config.dset.grid_dim),
        resolution=float(nf_config.dset.resolution),
    )
    if atoms_grid is not None:
        atoms_centered = {
            "coords": (
                atoms_grid["coords"]
                - (int(nf_config.dset.grid_dim) - 1) / 2.0
            )
            * float(nf_config.dset.resolution),
            "atoms_channel": atoms_grid["atoms_channel"],
        }
    else:
        atoms_centered = None
    _atoms_dump(condition_dir / "generated_atoms.npz", atoms_centered)

    try:
        generated = decoder.codes_to_molecules_batched(
            unnormalized,
            unnormalize=False,
            config=nf_config,
            fabric=fabric,
            batch_size_render_codes=1,
            verbose=False,
            target_dirname=str(condition_dir),
        )
        filter_mol_to_sdf_pdb(
            nf_config,
            fabric,
            generated,
            center_coords=batch["receptor"]["center_coords"].detach().cpu(),
            fname="generated",
            dirname=str(condition_dir),
            verbose=False,
        )
        metrics["sdf_export"] = str(condition_dir / "generated.sdf")
    except Exception as error:
        metrics["sdf_export_error"] = repr(error)
        fabric.print(f">> warning: {name} SDF export failed: {error}")

    del grid, prediction_flat, unnormalized
    torch.cuda.empty_cache()
    return metrics


@hydra.main(
    config_path="configs",
    config_name="holo_density_fusion",
    version_base=None,
)
def main(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    output_dir = _resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(config.seed))
    fabric = setup_fabric(config)
    OmegaConf.save(config, output_dir / "requested_config.yaml")

    (
        model,
        ligand_encoder,
        decoder,
        model_config,
        nf_config,
        num_classes,
        fb_checkpoint,
        nf_checkpoint,
    ) = _load_models(config, fabric)
    OmegaConf.save(model_config, output_dir / "effective_funcbind_config.yaml")
    OmegaConf.save(nf_config, output_dir / "effective_nf_config.yaml")

    target_index = int(config.density.target_index)
    batch, test_size = _make_target_batch(nf_config, target_index, fabric)
    field_maker, field_maker_receptor = create_field_makers(
        model_config, nf_config, fabric
    )
    ligand_target = _ligand_code(
        batch, ligand_encoder, field_maker, model
    )
    with torch.no_grad():
        receptor_base = get_receptor_encoding(
            batch["receptor"],
            model,
            field_maker_receptor,
            fabric,
            rand_rots=None,
            n_chains=1,
        ).detach()
    label = get_label(batch, model_config, num_classes=num_classes)

    density_encoder, density_checkpoint = _load_density_encoder(config, fabric)
    crop_path = _resolve_path(config.density.crop_path)
    holo_density_np = np.load(crop_path).astype(np.float32)
    if holo_density_np.shape != (64, 64, 64):
        raise ValueError(f"expected 64^3 density crop, got {holo_density_np.shape}")

    apo_config = config.density.apo_mask
    apo_mask_np = None
    if bool(apo_config.enabled):
        fabric.print(">> masking ligand density with the ligand occupancy field")
        density_np, apo_mask_np, mask_report = _apply_apo_mask(
            holo_density_np,
            batch,
            apo_config,
            grid_dim=64,
            resolution=0.25,
            device=fabric.device,
        )
        fabric.print(
            f">> apo mask: density at ligand atoms "
            f"{mask_report['ligand_atom_density_before']:.3f} -> "
            f"{mask_report['ligand_atom_density_after']:.3f}; "
            f"envelope above {mask_report['leakage_sigma']} sigma "
            f"{mask_report['envelope_above_sigma_before']:.3f} -> "
            f"{mask_report['envelope_above_sigma_after']:.3f}"
        )
        density_arm = "apo_density_funcbind"
    else:
        density_np = holo_density_np
        mask_report = {"enabled": False}
        density_arm = "holo_density_funcbind"

    density_input = _build_density_input(
        batch,
        density_np,
        _resolve_path(config.density.voxbind_python_root),
        fabric,
    )
    with torch.no_grad(), fabric.autocast():
        tokens = density_encoder.forward_features(density_input)
        pooled = _unwrap(density_encoder)._pool_groups(tokens)
        density_feature = (
            pooled.transpose(1, 2)
            .reshape(1, 512, 8, 8, 8)
            .detach()
        )
    del tokens, pooled, density_input, density_encoder
    torch.cuda.empty_cache()

    adapter = DensityConditionAdapter()
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config.density.learning_rate),
        weight_decay=float(config.density.weight_decay),
    )
    adapter, optimizer = fabric.setup(adapter, optimizer)
    receptor_fused, history = _fit_adapter(
        adapter,
        optimizer,
        density_feature,
        receptor_base,
        ligand_target,
        label,
        model,
        config,
        fabric,
    )
    torch.save(
        {
            "adapter_state_dict": _unwrap(adapter).state_dict(),
            "history": history,
            "target_index": target_index,
        },
        output_dir / "density_adapter.pt",
    )
    _atomic_json(output_dir / "training_history.json", history)

    # The transfer arm reuses an adapter trained on a different density (the
    # holo map) and feeds it this run's density feature, so a change in the
    # result can only come from the density input. Restoring the RNG state
    # around it keeps y_init identical to a run without the transfer arm.
    transfer_receptor = None
    transfer_adapter_path = config.density.transfer_adapter_path
    if transfer_adapter_path is not None:
        transfer_adapter_path = _resolve_path(transfer_adapter_path)
        saved_cpu_state = torch.get_rng_state()
        saved_cuda_state = torch.cuda.get_rng_state()
        fabric.print(f">> transfer arm: loading adapter {transfer_adapter_path}")
        transfer_adapter = DensityConditionAdapter()
        transfer_adapter.load_state_dict(
            torch.load(transfer_adapter_path, map_location="cpu")[
                "adapter_state_dict"
            ],
            strict=True,
        )
        for parameter in transfer_adapter.parameters():
            parameter.requires_grad_(False)
        transfer_adapter.eval()
        transfer_adapter = fabric.setup_module(transfer_adapter)
        with torch.no_grad(), fabric.autocast():
            transfer_receptor = (
                receptor_base + transfer_adapter(density_feature)
            ).detach()
        del transfer_adapter
        torch.cuda.empty_cache()
        torch.set_rng_state(saved_cpu_state)
        torch.cuda.set_rng_state(saved_cuda_state)

    n_chains = int(config.sampling.n_chains)
    receptor_base_n = receptor_base.repeat(n_chains, 1, 1, 1, 1)
    receptor_fused_n = receptor_fused.repeat(n_chains, 1, 1, 1, 1)
    label_n = label.repeat(n_chains, 1) if label is not None else None
    initial_distribution = UniformPlusNormal(
        _unwrap(model).code_stats,
        model_config,
        device=fabric.device,
        dtype=torch.float32,
    )
    y_init = initial_distribution.sample()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    arms = [
        ("original_funcbind", "Original FuncBind", receptor_base_n),
        (density_arm, _ARM_LABELS[density_arm], receptor_fused_n),
    ]
    if transfer_receptor is not None:
        arms.append(
            (
                "transfer_adapter_funcbind",
                _ARM_LABELS["transfer_adapter_funcbind"],
                transfer_receptor.repeat(n_chains, 1, 1, 1, 1),
            )
        )

    ligand_target_cpu = ligand_target.float().cpu()
    codes: dict[str, torch.Tensor] = {}
    latent_mse: dict[str, torch.Tensor] = {}
    best_index: dict[str, int] = {}
    for key, arm_label, receptor_encoding in arms:
        fabric.print(f">> paired sampling: {arm_label}")
        sampled = _sample_codes(
            model,
            receptor_encoding,
            label_n,
            y_init,
            model_config,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
        )
        codes[key] = sampled
        latent_mse[key] = (
            (sampled - ligand_target_cpu).square().flatten(1).mean(1)
        )
        best_index[key] = int(latent_mse[key].argmin())

    latents = {
        "target_code": ligand_target_cpu,
        "y_init": y_init.detach().cpu(),
        "arms": [key for key, _, _ in arms],
        "codes": codes,
        "latent_mse": latent_mse,
        "best_index": best_index,
    }
    # Legacy key names kept so existing readers of the holo run still work.
    latents["original_codes"] = codes["original_funcbind"]
    latents["latent_mse_original"] = latent_mse["original_funcbind"]
    latents["original_best_index"] = best_index["original_funcbind"]
    torch.save(latents, output_dir / "sampled_latents.pt")

    query_points = batch["ligand"]["xs"]
    with torch.no_grad():
        target_flat = field_maker.compute_occupancies(
            batch["ligand"], num_channels=len(LIGAND_ELEMENTS)
        ).float().cpu()
    _atomic_sparse_occupancy_dump(
        output_dir / "reference_occupancy_points.npz",
        query_points=query_points,
        occupancy=target_flat,
        threshold=float(config.density.occupancy_point_threshold),
        grid_dim=int(nf_config.dset.grid_dim),
        resolution=float(nf_config.dset.resolution),
    )
    valid = batch["ligand"]["atoms_channel"][0] < len(LIGAND_ELEMENTS)
    reference_atoms = {
        "coords": batch["ligand"]["coords"][:, valid],
        "atoms_channel": batch["ligand"]["atoms_channel"][:, valid],
    }
    _atoms_dump(output_dir / "reference_atoms.npz", reference_atoms)
    # Named for the role, not the state: this is whatever map was fed to the
    # encoder, which is the apo (ligand-masked) map when apo_mask is enabled.
    np.savez_compressed(
        output_dir / "density_grid.npz",
        density=density_np.astype(np.float16),
        resolution=np.asarray(0.25, dtype=np.float32),
    )
    _density_cloud(
        output_dir / "density_points.npz",
        density_np,
        float(config.density.density_point_threshold),
    )
    if apo_mask_np is not None:
        np.savez_compressed(
            output_dir / "apo_mask.npz",
            mask=apo_mask_np.astype(np.float16),
            holo_density=holo_density_np.astype(np.float16),
            apo_density=density_np.astype(np.float16),
            resolution=np.asarray(0.25, dtype=np.float32),
        )

    results = {
        key: _evaluate_code(
            key,
            codes[key][best_index[key] : best_index[key] + 1],
            decoder,
            batch,
            target_flat,
            query_points,
            output_dir,
            config,
            nf_config,
            fabric,
        )
        for key, _, _ in arms
    }

    limitations = [
        "The density adapter is overfit on this same single target.",
        "Best-of-chain visualization uses target latent MSE as an oracle.",
        "This is a proof of concept, not a held-out density-fusion evaluation.",
    ]
    if apo_mask_np is None:
        limitations.insert(
            0, "The holo map contains the crystallographic reference ligand."
        )
    else:
        limitations.insert(
            0,
            "The apo map is a masked holo map, not a separately measured apo "
            "structure: solvent reorganization and side-chain relaxation on "
            "ligand release are not modeled, and density from receptor atoms "
            "that fall inside the ligand envelope is erased along with it.",
        )

    report = {
        "status": "complete",
        "experiment": (
            "single-target apo-density (ligand-masked) FuncBind adapter"
            if apo_mask_np is not None
            else "single-target holo-density-conditioned FuncBind adapter"
        ),
        "target": {
            "crossdocked_test_index": target_index,
            "test_set_size": test_size,
            "pdb_id": str(config.density.pdb_id),
            "ligand_id": str(config.density.ligand_id),
            "receptor_id": str(batch["receptor"]["id"][0]),
            "ligand_file_id": str(batch["ligand"]["id"][0]),
        },
        "checkpoints": {
            "funcbind": str(fb_checkpoint),
            "neural_field": str(nf_checkpoint),
            "density_encoder": str(density_checkpoint),
        },
        "density_input": {
            "crop": str(crop_path),
            "shape": list(density_np.shape),
            "min": float(density_np.min()),
            "max": float(density_np.max()),
            "mean": float(density_np.mean()),
            "std": float(density_np.std()),
            "channels": [
                "7 zero ligand atom channels (unknown during generation)",
                "4 receptor atom-blob channels",
                (
                    "aligned real 2Fo-Fc density with the ligand site erased"
                    if apo_mask_np is not None
                    else "aligned real holo 2Fo-Fc density"
                ),
                "z-scored density gradient magnitude",
            ],
            "feature_shape": _tensor_shape(density_feature),
        },
        "apo_mask": mask_report,
        "adapter": {
            "trainable_parameters": sum(
                p.numel() for p in _unwrap(adapter).parameters()
            ),
            "overfit_steps": int(config.density.overfit_steps),
            "final_training_record": history[-1],
            "transfer_adapter": (
                str(transfer_adapter_path)
                if transfer_receptor is not None
                else None
            ),
            "frozen_components": [
                "FuncBind receptor encoder",
                "FuncBind diffusion denoiser",
                "neural-field ligand encoder",
                "original INR decoder",
                "VoxBind ChannelViT density encoder",
            ],
        },
        "arms": [
            {"key": key, "label": arm_label} for key, arm_label, _ in arms
        ],
        "sampling": {
            "paired_initial_latents": True,
            "paired_sampler_rng": True,
            "n_chains": n_chains,
            "sampler_steps": int(model_config.sampler.N),
            "latent_mse": {
                key: latent_mse[key].tolist() for key in latent_mse
            },
            "best_index": best_index,
            "original_latent_mse": latent_mse["original_funcbind"].tolist(),
            "original_best_index": best_index["original_funcbind"],
            "selection": "oracle minimum latent MSE to the target code, separately per method",
        },
        "results": results,
        "limitations": limitations,
    }
    _atomic_json(output_dir / "result.json", report)
    fabric.print(f">> complete: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
