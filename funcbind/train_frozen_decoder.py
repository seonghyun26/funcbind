"""Train one decoder over small molecules using a frozen FuncBind encoder."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from funcbind.compare_decoders import (
    _atomic_json_dump,
    _decoder_config,
    _parameter_count,
    atom_recovery_metrics,
    reconstruction_metrics,
)
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.dataset.field_maker import FieldMaker
from funcbind.models.decoder import get_atom_coords_batched, get_code_spatial
from funcbind.models.encoder import sample_posterior
from funcbind.reconstruct_all_targets import aggregate_results
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_dataset import RandomizedMinorityUpsampler
from funcbind.utils.utils_nf import create_nf_decoder, create_nf_encoder


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, relative_to / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_id(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable_id(item) for item in value]
    return str(value)


def _load_frozen_encoder(
    config: DictConfig,
    fabric,
    checkpoint_path: Path,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("enc_state_dict")
    if state is None:
        raise KeyError(f"{checkpoint_path} has no enc_state_dict")
    encoder = create_nf_encoder(config, fabric)
    encoder.load_state_dict(state, strict=True)
    del checkpoint, state
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return fabric.setup_module(encoder)


@torch.no_grad()
def _cache_latent_codes(
    config: DictConfig,
    fabric,
    encoder,
    dataset: DatasetOmni,
) -> torch.Tensor:
    field_maker = FieldMaker(config).to(fabric.device)
    codes = []
    for index in range(len(dataset)):
        _seed(int(config.seed) + index)
        batch = fabric.to_device(collate_fn([dataset[index]]))
        voxels = field_maker.compute_voxel_grid(
            batch["ligand"], num_channels=int(config.dset.n_channels)
        )
        with fabric.autocast():
            moments = encoder(voxels)
            latent, _ = sample_posterior(
                moments,
                sample_posterior=False,
                save_log_var=False,
                deterministic=True,
            )
        codes.append(latent.detach().float().cpu())
        del batch, voxels, moments, latent
    return torch.cat(codes, dim=0)


@torch.no_grad()
def _encode_batch(config, fabric, encoder, field_maker, batch) -> torch.Tensor:
    """Encode the very batch the target is built from.

    The cached path in :func:`_cache_latent_codes` draws each sample under
    ``seed(seed + index)`` while the training loop redraws it under
    ``seed(seed + epoch * len + index)``. With ``data_aug`` on those are two
    different random poses, so the cached latent describes a different rotation
    than the target. Encoding inline removes that skew, and removes the
    ``len(dataset) x 2 MiB`` host-RAM cache with it.
    """
    voxels = field_maker.compute_voxel_grid(
        batch["ligand"], num_channels=int(config.dset.n_channels)
    )
    with fabric.autocast():
        moments = encoder(voxels)
        latent, _ = sample_posterior(
            moments,
            sample_posterior=False,
            save_log_var=False,
            deterministic=True,
        )
    return latent.detach().float()


_POSE_KEYS = ("coords", "atoms_channel", "radius")


@torch.no_grad()
def _build_latent_pose_cache(
    config: DictConfig,
    fabric,
    encoder,
    field_maker,
    dataset: DatasetOmni,
    indices: list[int],
    window: int,
):
    """Encode one frozen pose per sample and keep it for a whole window.

    The encoder is frozen, so re-running it every step is pure waste -- the
    original neural-field training batched it 16-32x and the ten-target splat fit
    encoded once and never again. Here each window draws one augmented pose per
    sample, encodes it, and stores the latent in fp16 alongside the ligand atoms
    that produced it. Training then rebuilds the target from those same atoms, so
    latent and target always describe one pose, while the query points stay fresh
    every epoch. Bumping the window redraws every pose, which is what keeps
    data_aug meaningful.
    """
    n = len(indices)
    # Allocated lazily from the first latent so the shape always matches the
    # encoder rather than a config value that could drift from it.
    latents = None
    poses: list[dict[str, torch.Tensor]] = []
    started = time.perf_counter()
    for position, index in enumerate(indices):
        _seed(int(config.seed) + window * len(dataset) + index)
        sample = dataset[index]
        batch = fabric.to_device(collate_fn([sample]))
        voxels = field_maker.compute_voxel_grid(
            batch["ligand"], num_channels=int(config.dset.n_channels)
        )
        with fabric.autocast():
            moments = encoder(voxels)
            latent, _ = sample_posterior(
                moments,
                sample_posterior=False,
                save_log_var=False,
                deterministic=True,
            )
        latent = latent.detach().half().cpu()
        if latents is None:
            latents = torch.empty((n, *latent.shape[1:]), dtype=torch.float16)
            fabric.print(
                f">> latent cache window {window}: allocating "
                f"{n:,} x {tuple(latent.shape[1:])} fp16 = "
                f"{n * latent[0].numel() * 2 / 1e9:.0f} GB"
            )
        latents[position] = latent[0]
        poses.append(
            {key: sample["ligand"][key].clone() for key in _POSE_KEYS}
        )
        del batch, voxels, moments, latent
        if position and position % 20000 == 0:
            rate = position / (time.perf_counter() - started)
            fabric.print(
                f">> latent cache window {window}: {position:,}/{n:,} "
                f"({rate:.1f}/s, eta {(n - position) / rate / 3600:.1f} h)"
            )
    fabric.print(
        f">> latent cache window {window}: {n:,} poses encoded in "
        f"{(time.perf_counter() - started) / 3600:.2f} h, "
        f"{latents.numel() * 2 / 1e9:.0f} GB fp16"
    )
    return latents, poses


def _target_from_pose(
    config: DictConfig,
    fabric,
    field_maker,
    dataset: DatasetOmni,
    pose: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh query points over a cached pose, plus the occupancies they imply."""
    ligand = {key: value.unsqueeze(0).to(fabric.device) for key, value in pose.items()}
    ligand["xs"] = dataset._get_xs(pose).unsqueeze(0).to(fabric.device)
    target = field_maker.compute_occupancies(
        ligand, num_channels=int(config.dset.n_channels)
    )
    return ligand["xs"], target


def _epoch_indices(config, dataset, epoch: int) -> list[int]:
    """Sample-order for one epoch.

    When the dataset carries clusters, reuse FuncBind's own
    :class:`RandomizedMinorityUpsampler`: it reshuffles every cluster and
    round-robins them, so each epoch draws a *different* subset of the larger
    modality. Indexing ``range(len(dataset))`` instead would truncate to a
    prefix and pin the same MCP conformers forever.
    """
    clusters = getattr(dataset, "cluster_dict", None)
    if clusters and bool(config.dset.get("rebalance", False)):
        _seed(int(config.seed) + epoch)
        return list(RandomizedMinorityUpsampler(clusters))
    return [
        int(value)
        for value in np.random.default_rng(
            int(config.seed) + epoch
        ).permutation(len(dataset))
    ]


def _forward_from_latent(
    decoder,
    query_points: torch.Tensor,
    latent_grid: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    codes = (
        latent_grid
        if variant == "gaussian_splat"
        else get_code_spatial(query_points, latent_grid)
    )
    return decoder(query_points, codes)


@torch.no_grad()
def _render_from_latent(
    decoder,
    query_points: torch.Tensor,
    latent_grid: torch.Tensor,
    *,
    variant: str,
    chunk_size: int,
) -> torch.Tensor:
    predictions = []
    decoder.eval()
    for query_chunk in query_points.split(chunk_size, dim=1):
        predictions.append(
            _forward_from_latent(
                decoder, query_chunk, latent_grid, variant
            )
            .float()
            .cpu()
        )
    return torch.cat(predictions, dim=1)


def _save_training_checkpoint(
    path: Path,
    *,
    decoder,
    optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    config: DictConfig,
    fabric,
) -> None:
    module = getattr(decoder, "module", decoder)
    fabric.save(
        path,
        {
            "decoder_state_dict": module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": config,
        },
    )


def _train_decoder(
    config: DictConfig,
    fabric,
    decoder,
    optimizer,
    latent_codes: torch.Tensor | None,
    dataset: DatasetOmni,
    output_dir: Path,
    encoder=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if latent_codes is None and encoder is None:
        raise ValueError("pass either cached latent_codes or a frozen encoder")
    field_maker = FieldMaker(config).to(fabric.device)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    checkpoint_path = output_dir / "decoder_model.pt"
    if bool(config.resume) and checkpoint_path.exists():
        checkpoint = fabric.load(checkpoint_path)
        module = getattr(decoder, "module", decoder)
        module.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        fabric.print(f">> resuming decoder at epoch {start_epoch}")

    if fabric.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fabric.device)
    step_times = []
    dataset.targeted_sampling_ratio = int(
        config.dset.targeted_sampling_ratio
    )
    decoder.train()
    refresh_every = int(config.get("latent_cache_refresh_every", 0) or 0)
    windowed = latent_codes is None and encoder is not None and refresh_every > 0
    window_latents = None
    window_poses = None
    window_indices = None
    current_window = None
    for epoch in range(start_epoch, int(config.epochs)):
        if windowed:
            window = epoch // refresh_every
            if window != current_window:
                # One pose per sample, held for the whole window. Indices are
                # frozen with it so the cache and the epoch order stay aligned.
                window_indices = _epoch_indices(config, dataset, window)
                window_latents = None  # release before allocating the next one
                gc.collect()
                window_latents, window_poses = _build_latent_pose_cache(
                    config, fabric, encoder, field_maker,
                    dataset, window_indices, window,
                )
                current_window = window
            indices = list(range(len(window_indices)))
        else:
            indices = _epoch_indices(config, dataset, epoch)
        # Query points are redrawn every epoch even when the pose is frozen,
        # otherwise every step inside a window would repeat identically.
        epoch_rng = np.random.default_rng(int(config.seed) + epoch)
        epoch_rng.shuffle(indices)
        epoch_squared_error = 0.0
        epoch_elements = 0
        intersection = 0
        union = 0
        for position, index_value in enumerate(indices):
            index = int(index_value)
            if windowed:
                _seed(int(config.seed) + epoch * len(dataset) + index)
                query_points, target = _target_from_pose(
                    config, fabric, field_maker, dataset, window_poses[index]
                )
                latent = window_latents[index : index + 1].to(
                    fabric.device, non_blocking=True
                ).float()
            else:
                _seed(
                    int(config.seed)
                    + epoch * len(dataset)
                    + index
                )
                batch = fabric.to_device(collate_fn([dataset[index]]))
                query_points = batch["ligand"]["xs"]
                target = field_maker.compute_occupancies(
                    batch["ligand"], num_channels=int(config.dset.n_channels)
                )
                if latent_codes is not None:
                    latent = latent_codes[index : index + 1].to(
                        fabric.device, non_blocking=True
                    )
                else:
                    # Same `batch`, so latent and target share one pose.
                    latent = _encode_batch(
                        config, fabric, encoder, field_maker, batch
                    )

            optimizer.zero_grad()
            if fabric.device.type == "cuda":
                torch.cuda.synchronize(fabric.device)
            started = time.perf_counter()
            prediction = _forward_from_latent(
                decoder, query_points, latent, str(config.variant)
            )
            loss = torch.nn.functional.mse_loss(prediction, target)
            fabric.backward(loss)
            if float(config.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), float(config.gradient_clip_norm)
                )
            optimizer.step()
            if fabric.device.type == "cuda":
                torch.cuda.synchronize(fabric.device)
            step_times.append(time.perf_counter() - started)

            epoch_squared_error += float(
                (prediction.detach().float() - target.float()).square().sum()
            )
            epoch_elements += prediction.numel()
            prediction_binary = prediction.detach() >= 0.5
            target_binary = target >= 0.5
            intersection += int(
                (prediction_binary & target_binary).sum()
            )
            union += int((prediction_binary | target_binary).sum())
            if not windowed:
                del batch
            del query_points, target, latent, prediction, loss

        metrics = {
            "epoch": epoch,
            "density_mse": epoch_squared_error / epoch_elements,
            "miou": intersection / union if union else 1.0,
        }
        history.append(metrics)
        _atomic_json_dump(
            {
                "status": "training",
                "variant": str(config.variant),
                "history": history,
            },
            output_dir / "training_progress.json",
        )
        fabric.print(
            f">> {config.variant} epoch {epoch + 1}/{config.epochs}: "
            f"mse={metrics['density_mse']:.4e}, "
            f"miou={metrics['miou']:.4f}"
        )
        if (
            (epoch + 1) % int(config.save_every_epochs) == 0
            or epoch + 1 == int(config.epochs)
        ):
            _save_training_checkpoint(
                checkpoint_path,
                decoder=decoder,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                config=config,
                fabric=fabric,
            )

    warmup = min(int(config.timing_warmup_steps), len(step_times))
    timed = step_times[warmup:] or step_times
    timing = {
        "mean_step_seconds": (
            float(np.mean(timed)) if timed else None
        ),
        "median_step_seconds": (
            float(np.median(timed)) if timed else None
        ),
        "peak_gpu_memory_gib": (
            float(torch.cuda.max_memory_allocated(fabric.device) / 1024**3)
            if fabric.device.type == "cuda"
            else None
        ),
    }
    return history, timing


def _evaluate_all_targets(
    config: DictConfig,
    fabric,
    decoder,
    latent_codes: torch.Tensor | None,
    dataset: DatasetOmni,
    output_dir: Path,
    encoder=None,
) -> dict[str, Any]:
    if latent_codes is None and encoder is None:
        raise ValueError("pass either cached latent_codes or a frozen encoder")
    field_maker = FieldMaker(config).to(fabric.device)
    target_dir = output_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    # Full-grid evaluation renders 128^3 query points per sample, so evaluating
    # a 286k-sample corpus is not viable; cap it and record the cap.
    limit = config.get("eval_max_targets", None)
    eval_indices = list(range(len(dataset)))
    if limit is not None and int(limit) < len(eval_indices):
        eval_indices = eval_indices[: int(limit)]
        fabric.print(
            f">> evaluating {len(eval_indices)}/{len(dataset)} targets "
            f"(eval_max_targets={int(limit)})"
        )
    for index in eval_indices:
        result_path = target_dir / f"target_{index:03d}.json"
        if bool(config.resume) and result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                if result.get("status") == "complete":
                    results.append(result)
                    continue
            except json.JSONDecodeError:
                pass
        try:
            batch = fabric.to_device(collate_fn([dataset[index]]))
            query_points = batch["ligand"]["xs"]
            if latent_codes is not None:
                latent = latent_codes[index : index + 1].to(fabric.device)
            else:
                latent = _encode_batch(
                    config, fabric, encoder, field_maker, batch
                )
            with torch.no_grad():
                target = field_maker.compute_occupancies(
                    batch["ligand"],
                    num_channels=int(config.dset.n_channels),
                ).cpu()
                prediction = _render_from_latent(
                    decoder,
                    query_points,
                    latent,
                    variant=str(config.variant),
                    chunk_size=int(config.evaluation_query_chunk_size),
                )
            grid_dim = int(config.dset.grid_dim)
            prediction_grid = prediction.permute(0, 2, 1).reshape(
                -1,
                int(config.dset.n_channels),
                grid_dim,
                grid_dim,
                grid_dim,
            )
            predicted_atoms = get_atom_coords_batched(
                prediction_grid.clone(),
                fabric,
                rad=float(config.dset.ligand_radius),
                resolution=float(config.dset.resolution),
                verbose=False,
                flexible=True,
            )[0]
            result = {
                "status": "complete",
                "sample_index": index,
                "target_id": _jsonable_id(batch["ligand"].get("id")),
                "final": {},
                "timing": {},
                "full_grid": {
                    "reconstruction": reconstruction_metrics(
                        prediction, target
                    ),
                    "atom_recovery": atom_recovery_metrics(
                        predicted_atoms,
                        batch["ligand"],
                        grid_dim=grid_dim,
                        resolution=float(config.dset.resolution),
                        match_distance=float(config.atom_match_distance),
                    ),
                },
            }
            _atomic_json_dump(result, result_path)
            results.append(result)
            del (
                batch,
                query_points,
                latent,
                target,
                prediction,
                prediction_grid,
                predicted_atoms,
            )
        except Exception as exc:
            failure = {
                "sample_index": index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _atomic_json_dump(
                failure, target_dir / f"target_{index:03d}_error.json"
            )
        finally:
            gc.collect()
            if fabric.device.type == "cuda":
                torch.cuda.empty_cache()

        summary = aggregate_results(
            results,
            variant=str(config.variant),
            dataset_size=len(eval_indices),
            requested_indices=eval_indices,
            failures=failures,
        )
        _atomic_json_dump(summary, output_dir / "reconstruction_summary.json")
        fabric.print(
            f">> evaluated {index + 1}/{len(dataset)} targets "
            f"({len(failures)} failures)"
        )

    return aggregate_results(
        results,
        variant=str(config.variant),
        dataset_size=len(eval_indices),
        requested_indices=eval_indices,
        failures=failures,
    )


@hydra.main(
    config_path="configs",
    config_name="train_frozen_decoder",
    version_base=None,
)
def main(config: DictConfig) -> None:
    variant = str(config.variant)
    if variant not in {"inr", "gaussian_splat"}:
        raise ValueError("variant must be 'inr' or 'gaussian_splat'")
    package_root = Path(__file__).resolve().parent
    data_dir = _resolve_path(
        str(config.dset.data_dir), relative_to=package_root
    )
    config.dset.data_dir = str(data_dir)
    checkpoint_path = _resolve_path(
        str(config.pretrained_encoder_checkpoint),
        relative_to=package_root.parent,
    )
    output_dir = Path(str(config.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    fabric = setup_fabric(config)
    cache_latents = bool(config.get("cache_latents", True))
    # Rebalancing needs the dataset to build cluster_dict; it also changes
    # len(dataset) to min(cluster) * n_clusters.
    rebalance = bool(config.dset.get("rebalance", False))
    training_dataset = DatasetOmni(
        config,
        split=str(config.split),
        sample_points=True,
        sample_full_grid=False,
        rebalance=rebalance,
    )
    full_grid_dataset = DatasetOmni(
        config,
        split=str(config.get("eval_split", config.split)),
        sample_points=True,
        sample_full_grid=True,
        rebalance=False,
    )

    encoder = _load_frozen_encoder(config, fabric, checkpoint_path)
    latent_codes = None
    if cache_latents:
        if rebalance:
            raise ValueError(
                "cache_latents indexes self.data positionally, which the "
                "rebalanced index order breaks; set cache_latents=false"
            )
        if bool(config.dset.get("data_aug", False)):
            raise ValueError(
                "cache_latents encodes each sample under a different seed than "
                "the training loop redraws it with, so data_aug would pair a "
                "latent and a target from two different poses; set "
                "cache_latents=false or data_aug=false"
            )
        encoding_dataset = DatasetOmni(
            config,
            split=str(config.split),
            sample_points=False,
            sample_full_grid=False,
            rebalance=False,
        )
        latent_codes = _cache_latent_codes(
            config, fabric, encoder, encoding_dataset
        )
        del encoder, encoding_dataset
        encoder = None
        gc.collect()
        if fabric.device.type == "cuda":
            torch.cuda.empty_cache()
        fabric.print(
            f">> cached deterministic pretrained latents: "
            f"{tuple(latent_codes.shape)}"
        )
    else:
        fabric.print(
            ">> encoding inline from the training batch (no latent cache)"
        )

    variant_config = _decoder_config(config, variant)
    _seed(int(config.seed))
    decoder = create_nf_decoder(variant_config, fabric)
    decoder_parameters = _parameter_count(decoder)
    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=float(config.lr_decoder)
    )
    decoder, optimizer = fabric.setup(decoder, optimizer)
    history, timing = _train_decoder(
        variant_config,
        fabric,
        decoder,
        optimizer,
        latent_codes,
        training_dataset,
        output_dir,
        encoder=encoder,
    )
    evaluation = _evaluate_all_targets(
        variant_config,
        fabric,
        decoder,
        latent_codes,
        full_grid_dataset,
        output_dir,
        encoder=encoder,
    )
    final = {
        "status": (
            "complete"
            if evaluation["failure_count"] == 0
            else "complete_with_failures"
        ),
        "variant": variant,
        "pretrained_encoder_checkpoint": str(checkpoint_path),
        "encoder_frozen": True,
        "decoder_initialization": "fresh",
        "decoder_parameters": decoder_parameters,
        "training_history": history,
        "timing": timing,
        "reconstruction": evaluation,
    }
    _atomic_json_dump(final, output_dir / "result.json")
    fabric.print(f">> result written to {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
