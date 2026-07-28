"""Single-sample INR versus Gaussian-splat decoder ablation.

This is intentionally separate from dataset-scale training.  It gives both
architectures the same sampled molecule, query points, occupancy targets, and
encoder initialization, then records reconstruction quality, step time, and GPU
memory.  Optional full-grid evaluation adds atom recovery, coordinate RMSD, and
element classification metrics after peak extraction.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.dataset.field_maker import FieldMaker
from funcbind.models.decoder import get_atom_coords_batched, get_code_spatial
from funcbind.utils.constants import PADDING_INDEX
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_nf import create_nf_decoder, create_nf_encoder


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _atomic_sparse_occupancy_dump(
    path: Path,
    *,
    query_points: torch.Tensor,
    occupancy: torch.Tensor,
    threshold: float,
    grid_dim: int,
    resolution: float,
) -> None:
    """Save occupied full-grid points without retaining a dense 128³ tensor."""
    if query_points.shape[0] != 1 or occupancy.shape[0] != 1:
        raise ValueError("Sparse occupancy export currently requires batch size 1")
    if query_points.shape[1] != occupancy.shape[1]:
        raise ValueError("Query-point and occupancy point counts must match")

    point_indices, channels = torch.where(occupancy[0] >= threshold)
    scale_angstrom = resolution * grid_dim / 2.0
    coordinates = (
        query_points[0, point_indices].detach().float().cpu()
        * scale_angstrom
    )
    values = occupancy[0, point_indices, channels].detach().float().cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            coordinates=coordinates.numpy().astype(np.float32),
            channels=channels.detach().cpu().numpy().astype(np.int16),
            occupancies=values.numpy().astype(np.float16),
            threshold=np.asarray(threshold, dtype=np.float32),
            grid_dim=np.asarray(grid_dim, dtype=np.int16),
            resolution=np.asarray(resolution, dtype=np.float32),
        )
    os.replace(temporary, path)


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _decoder_config(config: DictConfig, name: str) -> DictConfig:
    variant = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    variant.decoder = copy.deepcopy(variant[f"{name}_decoder"])
    return variant


def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = prediction.detach().float()
    target = target.detach().float()
    reduction_dims = tuple(range(prediction.ndim - 1))
    channel_mse = (prediction - target).square().mean(dim=reduction_dims)
    pred_binary = prediction >= threshold
    target_binary = target >= threshold
    intersection = (pred_binary & target_binary).sum(dim=reduction_dims).float()
    union = (pred_binary | target_binary).sum(dim=reduction_dims).float()
    channel_iou = [
        float(intersection[index] / union[index]) if union[index] else None
        for index in range(union.numel())
    ]
    total_union = union.sum()
    miou = float(intersection.sum() / total_union) if total_union else 1.0
    return {
        "density_mse": float((prediction - target).square().mean()),
        "density_mae": float((prediction - target).abs().mean()),
        "miou": miou,
        "channel_mse": [float(value) for value in channel_mse],
        "channel_iou": channel_iou,
    }


def atom_recovery_metrics(
    predicted: dict[str, torch.Tensor] | None,
    target: dict[str, torch.Tensor],
    *,
    grid_dim: int,
    resolution: float,
    match_distance: float,
) -> dict[str, Any]:
    target_channels = target["atoms_channel"][0].detach().cpu()
    valid_target = target_channels != PADDING_INDEX
    target_coords = target["coords"][0].detach().cpu()[valid_target].float()
    target_channels = target_channels[valid_target].long()

    if predicted is None:
        return {
            "n_target_atoms": int(target_coords.shape[0]),
            "n_predicted_atoms": 0,
            "matched_atoms": 0,
            "atom_precision": 0.0,
            "atom_recall": 0.0,
            "atom_f1": 0.0,
            "coordinate_rmsd": None,
            "element_accuracy": None,
        }

    predicted_coords = predicted["coords"][0].detach().cpu().float()
    predicted_coords = (
        predicted_coords - (grid_dim - 1) / 2.0
    ) * resolution
    predicted_channels = predicted["atoms_channel"][0].detach().cpu().long()

    n_target = int(target_coords.shape[0])
    n_predicted = int(predicted_coords.shape[0])
    if n_target == 0 or n_predicted == 0:
        return {
            "n_target_atoms": n_target,
            "n_predicted_atoms": n_predicted,
            "matched_atoms": 0,
            "atom_precision": 0.0 if n_predicted else 1.0,
            "atom_recall": 0.0 if n_target else 1.0,
            "atom_f1": 0.0,
            "coordinate_rmsd": None,
            "element_accuracy": None,
        }

    distances = torch.cdist(predicted_coords, target_coords).numpy()
    pred_indices, target_indices = linear_sum_assignment(distances)
    assigned_distances = distances[pred_indices, target_indices]
    spatial_match = assigned_distances <= match_distance
    matched = int(spatial_match.sum())
    precision = matched / n_predicted
    recall = matched / n_target
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if matched:
        matched_distances = assigned_distances[spatial_match]
        coordinate_rmsd = float(np.sqrt(np.mean(matched_distances ** 2)))
        matched_pred = torch.from_numpy(pred_indices[spatial_match])
        matched_target = torch.from_numpy(target_indices[spatial_match])
        element_accuracy = float(
            (
                predicted_channels[matched_pred]
                == target_channels[matched_target]
            ).float().mean()
        )
    else:
        coordinate_rmsd = None
        element_accuracy = None

    same_element_matches = 0
    for channel in torch.unique(target_channels).tolist():
        pred_mask = torch.where(predicted_channels == channel)[0]
        target_mask = torch.where(target_channels == channel)[0]
        if pred_mask.numel() == 0 or target_mask.numel() == 0:
            continue
        channel_distances = distances[np.ix_(pred_mask.numpy(), target_mask.numpy())]
        row_indices, column_indices = linear_sum_assignment(channel_distances)
        same_element_matches += int(
            (channel_distances[row_indices, column_indices] <= match_distance).sum()
        )

    element_precision = same_element_matches / n_predicted
    element_recall = same_element_matches / n_target
    return {
        "n_target_atoms": n_target,
        "n_predicted_atoms": n_predicted,
        "matched_atoms": matched,
        "atom_precision": precision,
        "atom_recall": recall,
        "atom_f1": f1,
        "coordinate_rmsd": coordinate_rmsd,
        "element_accuracy": element_accuracy,
        "element_aware_precision": element_precision,
        "element_aware_recall": element_recall,
    }


def _forward(
    encoder,
    decoder,
    voxels: torch.Tensor,
    query_points: torch.Tensor,
    decoder_type: str,
) -> torch.Tensor:
    latent_grid = encoder(voxels)
    codes = (
        latent_grid
        if decoder_type == "gaussian_splat"
        else get_code_spatial(query_points, latent_grid)
    )
    return decoder(query_points, codes)


@torch.no_grad()
def _render_full_grid(
    encoder,
    decoder,
    voxels: torch.Tensor,
    query_points: torch.Tensor,
    *,
    decoder_type: str,
    query_chunk_size: int,
) -> torch.Tensor:
    encoder.eval()
    decoder.eval()
    latent_grid = encoder(voxels)
    predictions = []
    for query_chunk in query_points.split(query_chunk_size, dim=1):
        codes = (
            latent_grid
            if decoder_type == "gaussian_splat"
            else get_code_spatial(query_chunk, latent_grid)
        )
        predictions.append(decoder(query_chunk, codes).float().cpu())
    return torch.cat(predictions, dim=1)


def _make_batch(config: DictConfig, fabric, *, full_grid: bool):
    dataset = DatasetOmni(
        config,
        split=config.split,
        sample_points=True,
        sample_full_grid=full_grid,
        rebalance=False,
    )
    index = int(config.sample_index)
    if not 0 <= index < len(dataset):
        raise IndexError(f"sample_index {index} is outside dataset of size {len(dataset)}")
    return fabric.to_device(collate_fn([dataset[index]]))


def _train_variant(
    name: str,
    config: DictConfig,
    fabric,
    batch: dict[str, Any],
    initial_encoder_state: dict[str, torch.Tensor],
    output_dir: Path,
    full_grid_batch: dict[str, Any] | None,
) -> dict[str, Any]:
    variant_config = _decoder_config(config, name)
    decoder_type = str(variant_config.decoder.type)
    fabric.seed_everything(int(config.seed))

    encoder = create_nf_encoder(variant_config, fabric)
    encoder.load_state_dict(initial_encoder_state)
    decoder = create_nf_decoder(variant_config, fabric)
    encoder_parameters = _parameter_count(encoder)
    decoder_parameters = _parameter_count(decoder)

    train_encoder = bool(config.train_encoder)
    if not train_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    optimizer_encoder = None
    if train_encoder:
        optimizer_encoder = torch.optim.Adam(
            encoder.parameters(), lr=float(config.lr_encoder)
        )
        encoder, optimizer_encoder = fabric.setup(encoder, optimizer_encoder)
    else:
        encoder = fabric.setup_module(encoder)
    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=float(config.lr_decoder)
    )
    decoder, optimizer_decoder = fabric.setup(decoder, optimizer_decoder)

    field_maker = FieldMaker(variant_config).to(fabric.device)
    query_points = batch["ligand"]["xs"]
    with torch.no_grad():
        voxels = field_maker.compute_voxel_grid(
            batch["ligand"], num_channels=variant_config.dset.n_channels
        )
        target = field_maker.compute_occupancies(
            batch["ligand"], num_channels=variant_config.dset.n_channels
        )
        initial_prediction = _forward(
            encoder, decoder, voxels, query_points, decoder_type
        )
    initial_metrics = reconstruction_metrics(initial_prediction, target)

    partial_path = output_dir / f"{name}_partial.json"
    history = [{"step": 0, **initial_metrics}]
    partial = {
        "decoder": name,
        "decoder_type": decoder_type,
        "status": "running",
        "history": history,
    }
    _atomic_json_dump(partial, partial_path)

    if fabric.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fabric.device)
    step_times = []
    steps = int(config.steps)
    report_every = max(int(config.report_every), 1)
    encoder.train(train_encoder)
    decoder.train()
    for step in range(1, steps + 1):
        optimizer_decoder.zero_grad()
        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad()
        _synchronize(fabric.device)
        started = time.perf_counter()
        prediction = _forward(
            encoder, decoder, voxels, query_points, decoder_type
        )
        loss = torch.nn.functional.mse_loss(prediction, target)
        fabric.backward(loss)
        if float(config.gradient_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), float(config.gradient_clip_norm)
            )
            if train_encoder:
                torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), float(config.gradient_clip_norm)
                )
        if optimizer_encoder is not None:
            optimizer_encoder.step()
        optimizer_decoder.step()
        _synchronize(fabric.device)
        step_times.append(time.perf_counter() - started)

        if step % report_every == 0 or step == steps:
            with torch.no_grad():
                prediction = _forward(
                    encoder, decoder, voxels, query_points, decoder_type
                )
            metrics = reconstruction_metrics(prediction, target)
            history.append({"step": step, **metrics})
            partial["history"] = history
            partial["last_completed_step"] = step
            _atomic_json_dump(partial, partial_path)
            fabric.print(
                f">> {name} step {step}/{steps}: "
                f"mse={metrics['density_mse']:.4e}, miou={metrics['miou']:.4f}"
            )

    final_metrics = history[-1]
    warmup = min(int(config.timing_warmup_steps), max(len(step_times) - 1, 0))
    timed_steps = step_times[warmup:] or step_times
    timing = {
        "mean_step_seconds": float(np.mean(timed_steps)) if timed_steps else None,
        "median_step_seconds": float(np.median(timed_steps)) if timed_steps else None,
    }
    if fabric.device.type == "cuda":
        timing["peak_gpu_memory_gib"] = float(
            torch.cuda.max_memory_allocated(fabric.device) / (1024 ** 3)
        )
    else:
        timing["peak_gpu_memory_gib"] = None

    full_grid = None
    if full_grid_batch is not None:
        full_query_points = full_grid_batch["ligand"]["xs"]
        with torch.no_grad():
            full_voxels = field_maker.compute_voxel_grid(
                full_grid_batch["ligand"],
                num_channels=variant_config.dset.n_channels,
            )
            full_target = field_maker.compute_occupancies(
                full_grid_batch["ligand"],
                num_channels=variant_config.dset.n_channels,
            ).cpu()
            full_prediction = _render_full_grid(
                encoder,
                decoder,
                full_voxels,
                full_query_points,
                decoder_type=decoder_type,
                query_chunk_size=int(config.evaluation_query_chunk_size),
            )
        grid_dim = int(variant_config.dset.grid_dim)
        prediction_grid = full_prediction.permute(0, 2, 1).reshape(
            -1,
            int(variant_config.dset.n_channels),
            grid_dim,
            grid_dim,
            grid_dim,
        )
        predicted_atoms = get_atom_coords_batched(
            prediction_grid.clone(),
            fabric,
            rad=float(variant_config.dset.ligand_radius),
            resolution=float(variant_config.dset.resolution),
            verbose=False,
            flexible=True,
        )[0]
        full_grid = {
            "reconstruction": reconstruction_metrics(
                full_prediction, full_target
            ),
            "atom_recovery": atom_recovery_metrics(
                predicted_atoms,
                full_grid_batch["ligand"],
                grid_dim=grid_dim,
                resolution=float(variant_config.dset.resolution),
                match_distance=float(config.atom_match_distance),
            ),
        }
        if bool(config.get("save_occupancy_points", False)):
            threshold = float(config.get("occupancy_point_threshold", 0.1))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("occupancy_point_threshold must be in [0, 1]")
            _atomic_sparse_occupancy_dump(
                output_dir / f"{name}_occupancy_points.npz",
                query_points=full_query_points,
                occupancy=full_prediction,
                threshold=threshold,
                grid_dim=grid_dim,
                resolution=float(variant_config.dset.resolution),
            )
            reference_path = output_dir / "reference_occupancy_points.npz"
            if not reference_path.exists():
                _atomic_sparse_occupancy_dump(
                    reference_path,
                    query_points=full_query_points,
                    occupancy=full_target,
                    threshold=threshold,
                    grid_dim=grid_dim,
                    resolution=float(variant_config.dset.resolution),
                )
        del full_prediction, full_target, prediction_grid

    result = {
        "decoder": name,
        "decoder_type": decoder_type,
        "status": "complete",
        "encoder_parameters": encoder_parameters,
        "decoder_parameters": decoder_parameters,
        "initial": initial_metrics,
        "final": final_metrics,
        "timing": timing,
        "history": history,
        "full_grid": full_grid,
    }
    _atomic_json_dump(result, output_dir / f"{name}.json")
    partial_path.unlink(missing_ok=True)

    if bool(config.save_checkpoints):
        encoder_module = getattr(encoder, "module", encoder)
        decoder_module = getattr(decoder, "module", decoder)
        fabric.save(
            output_dir / f"{name}_model.pt",
            {
                "encoder_state_dict": encoder_module.state_dict(),
                "decoder_state_dict": decoder_module.state_dict(),
                "config": variant_config,
                "result": result,
            },
        )

    del encoder, decoder, optimizer_encoder, optimizer_decoder
    gc.collect()
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inr = results["inr"]
    gaussian = results["gaussian_splat"]
    delta = {
        "final_density_mse": (
            gaussian["final"]["density_mse"] - inr["final"]["density_mse"]
        ),
        "final_miou": gaussian["final"]["miou"] - inr["final"]["miou"],
        "mean_step_seconds": (
            gaussian["timing"]["mean_step_seconds"]
            - inr["timing"]["mean_step_seconds"]
        ),
    }
    gaussian_memory = gaussian["timing"]["peak_gpu_memory_gib"]
    inr_memory = inr["timing"]["peak_gpu_memory_gib"]
    delta["peak_gpu_memory_gib"] = (
        gaussian_memory - inr_memory
        if gaussian_memory is not None and inr_memory is not None
        else None
    )
    return {
        "interpretation": (
            "Gaussian-minus-INR: negative MSE/time/memory and positive mIoU "
            "favor Gaussian splatting."
        ),
        "gaussian_minus_inr": delta,
    }


@hydra.main(
    config_path="configs", config_name="compare_decoders", version_base=None
)
def main(config: DictConfig) -> None:
    if config.inr_decoder.code_dim != config.gaussian_splat_decoder.code_dim:
        raise ValueError("Both decoders must use the same code_dim")
    if config.reg_weight != 0:
        raise ValueError(
            "The controlled overfit comparison currently requires reg_weight=0"
        )

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    fabric = setup_fabric(config)
    batch = _make_batch(config, fabric, full_grid=False)
    full_grid_batch = (
        _make_batch(config, fabric, full_grid=True)
        if bool(config.full_grid_metrics)
        else None
    )

    base_config = _decoder_config(config, "inr")
    fabric.seed_everything(int(config.seed))
    initial_encoder = create_nf_encoder(base_config, fabric)
    initial_encoder_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_encoder.state_dict().items()
    }
    del initial_encoder

    results = {}
    for name in ("inr", "gaussian_splat"):
        results[name] = _train_variant(
            name,
            config,
            fabric,
            batch,
            initial_encoder_state,
            output_dir,
            full_grid_batch,
        )
        combined = {
            "status": "running" if len(results) == 1 else "complete",
            "sample_index": int(config.sample_index),
            "split": str(config.split),
            "results": results,
        }
        if len(results) == 2:
            combined["comparison"] = _comparison(results)
        _atomic_json_dump(combined, output_dir / "comparison.json")

    fabric.print(f">> decoder comparison written to {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
