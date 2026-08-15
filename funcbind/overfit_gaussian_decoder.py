"""Maximize same-target Gaussian-decoder fit on a frozen FuncBind encoder."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from funcbind.compare_decoders import (
    _atomic_json_dump,
    _atomic_sparse_occupancy_dump,
    _parameter_count,
    atom_recovery_metrics,
    reconstruction_metrics,
)
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.dataset.field_maker import FieldMaker
from funcbind.models.decoder import get_atom_coords_batched
from funcbind.models.encoder import sample_posterior
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_nf import create_nf_decoder, create_nf_encoder


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
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


def _draw_indices(
    pool: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if pool.numel() == 0:
        raise ValueError("cannot sample from an empty occupancy stratum")
    positions = torch.randint(
        pool.numel(), (count,), generator=generator, dtype=torch.long
    )
    return pool[positions]


def _stratified_indices(
    pools: dict[str, torch.Tensor],
    total: int,
    positive_fraction: float,
    tail_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, slice]]:
    n_positive = int(round(total * positive_fraction))
    n_tail = int(round(total * tail_fraction))
    n_background = total - n_positive - n_tail
    if min(n_positive, n_tail, n_background) <= 0:
        raise ValueError("every occupancy stratum must receive at least one point")
    pieces = {
        "positive": _draw_indices(pools["positive"], n_positive, generator),
        "tail": _draw_indices(pools["tail"], n_tail, generator),
        "background": _draw_indices(
            pools["background"], n_background, generator
        ),
    }
    indices = torch.cat(list(pieces.values()))
    slices = {}
    start = 0
    for name, values in pieces.items():
        slices[name] = slice(start, start + values.numel())
        start += values.numel()
    return indices, slices


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_weight: float,
) -> torch.Tensor:
    weights = 1.0 + target_weight * target
    return (weights * (prediction - target).square()).mean()


def _balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slices: dict[str, slice],
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    components = {}
    tensors = {}
    for name in ("positive", "tail", "background"):
        value = _weighted_mse(
            prediction[:, slices[name]],
            target[:, slices[name]],
            float(config.target_value_weight),
        )
        tensors[name] = value
        components[name] = float(value.detach())
    loss = (
        float(config.positive_loss_weight) * tensors["positive"]
        + float(config.tail_loss_weight) * tensors["tail"]
        + float(config.background_loss_weight) * tensors["background"]
    )
    components["total"] = float(loss.detach())
    return loss, components


@torch.no_grad()
def _render_full_grid(
    decoder,
    latent_grid: torch.Tensor,
    query_points: torch.Tensor,
    *,
    outer_chunk_size: int,
) -> torch.Tensor:
    decoder.eval()
    module = getattr(decoder, "module", decoder)
    parameters = module.decode_parameters(latent_grid)
    predictions = []
    for query_chunk in query_points.split(outer_chunk_size, dim=1):
        predictions.append(
            module._render_local(query_chunk, parameters).float().cpu()
        )
    return torch.cat(predictions, dim=1)


@hydra.main(
    config_path="configs",
    config_name="overfit_gaussian_decoder",
    version_base=None,
)
def main(config: DictConfig) -> None:
    package_root = Path(__file__).resolve().parent
    checkpoint_path = _resolve_path(
        str(config.pretrained_encoder_checkpoint),
        relative_to=package_root.parent,
    )
    data_dir = _resolve_path(str(config.data_dir), relative_to=package_root)
    output_dir = Path(str(config.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    _seed(int(config.seed))
    fabric = setup_fabric(config)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise KeyError(f"{checkpoint_path} has no config")
    model_config = OmegaConf.create(
        OmegaConf.to_container(checkpoint_config, resolve=True)
    )
    model_config.wandb = False
    model_config.n_devs = 1
    model_config.seed = int(config.seed)
    model_config.dirname = str(output_dir)
    model_config.dset.data_dir = str(data_dir)
    model_config.dset.data_aug = False
    model_config.dset.rebalance = False
    model_config.dset.batch_size = 1
    model_config.dset.num_workers = 0
    model_config.dset.use_single_dataset = str(config.dataset)
    model_config.decoder = copy.deepcopy(config.gaussian_splat_decoder)
    model_config.reg_weight = float(config.reg_weight)
    OmegaConf.save(model_config, output_dir / "model_config.yaml")
    OmegaConf.save(config, output_dir / "overfit_config.yaml")

    encoder = create_nf_encoder(model_config, fabric)
    encoder.load_state_dict(checkpoint["enc_state_dict"], strict=True)
    del checkpoint
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    encoder = fabric.setup_module(encoder)

    dataset = DatasetOmni(
        model_config,
        split=str(config.split),
        sample_points=True,
        sample_full_grid=True,
        rebalance=False,
    )
    sample_index = int(config.sample_index)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample_index {sample_index} outside dataset of size {len(dataset)}"
        )
    batch_cpu = collate_fn([dataset[sample_index]])
    batch_gpu = fabric.to_device(batch_cpu)
    field_maker = FieldMaker(model_config).to(fabric.device)
    full_query_gpu = batch_gpu["ligand"]["xs"]
    voxels = field_maker.compute_voxel_grid(
        batch_gpu["ligand"], num_channels=int(model_config.dset.n_channels)
    )
    full_target = field_maker.compute_occupancies(
        batch_gpu["ligand"], num_channels=int(model_config.dset.n_channels)
    ).float().cpu()
    with torch.no_grad(), fabric.autocast():
        moments = encoder(voxels)
        latent_grid, _ = sample_posterior(
            moments,
            sample_posterior=False,
            save_log_var=False,
            deterministic=True,
        )
    latent_grid = latent_grid.detach()
    full_query = full_query_gpu.float().cpu()
    del encoder, voxels, moments, batch_gpu
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()

    occupancy_max = full_target.amax(dim=-1)[0]
    positive_mask = occupancy_max >= float(config.positive_threshold)
    tail_mask = (
        (occupancy_max >= float(config.tail_threshold)) & ~positive_mask
    )
    background_mask = occupancy_max < float(config.tail_threshold)
    pools = {
        "positive": torch.where(positive_mask)[0],
        "tail": torch.where(tail_mask)[0],
        "background": torch.where(background_mask)[0],
    }
    fabric.print(
        ">> full-grid strata: "
        + ", ".join(f"{name}={values.numel():,}" for name, values in pools.items())
    )

    decoder = create_nf_decoder(model_config, fabric)
    decoder_parameters = _parameter_count(decoder)
    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=float(config.lr_decoder)
    )
    decoder, optimizer = fabric.setup(decoder, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.steps),
        eta_min=float(config.min_lr_decoder),
    )
    if fabric.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fabric.device)

    validation_generator = torch.Generator().manual_seed(
        int(config.seed) + 100_000
    )
    validation_indices, validation_slices = _stratified_indices(
        pools,
        int(config.validation_points),
        float(config.positive_fraction),
        float(config.tail_fraction),
        validation_generator,
    )
    validation_query = full_query[:, validation_indices].to(fabric.device)
    validation_target = full_target[:, validation_indices].to(fabric.device)
    training_generator = torch.Generator().manual_seed(int(config.seed) + 1)

    best_loss = float("inf")
    best_step = 0
    best_state = None
    history = []
    last_improvement = 0
    decoder.train()
    for step in range(1, int(config.steps) + 1):
        indices, slices = _stratified_indices(
            pools,
            int(config.batch_points),
            float(config.positive_fraction),
            float(config.tail_fraction),
            training_generator,
        )
        query = full_query[:, indices].to(fabric.device)
        target = full_target[:, indices].to(fabric.device)
        optimizer.zero_grad()
        prediction = decoder(query, latent_grid)
        loss, components = _balanced_loss(
            prediction, target, slices, config
        )
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), float(config.gradient_clip_norm)
        )
        optimizer.step()
        scheduler.step()

        if step % int(config.report_every) == 0 or step == 1:
            decoder.eval()
            with torch.no_grad():
                validation_prediction = decoder(
                    validation_query, latent_grid
                )
                validation_loss, validation_components = _balanced_loss(
                    validation_prediction,
                    validation_target,
                    validation_slices,
                    config,
                )
                validation_miou = reconstruction_metrics(
                    validation_prediction, validation_target
                )["miou"]
            validation_value = float(validation_loss)
            improved = validation_value < best_loss - float(config.min_delta)
            if improved:
                best_loss = validation_value
                best_step = step
                last_improvement = step
                module = getattr(decoder, "module", decoder)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            point = {
                "step": step,
                "lr": float(scheduler.get_last_lr()[0]),
                "training": components,
                "validation": validation_components,
                "validation_miou": validation_miou,
                "best_validation_loss": best_loss,
                "best_step": best_step,
            }
            history.append(point)
            _atomic_json_dump(
                {
                    "status": "training",
                    "history": history,
                    "best_step": best_step,
                    "best_validation_loss": best_loss,
                },
                output_dir / "training_progress.json",
            )
            fabric.print(
                f">> step {step}/{config.steps}: "
                f"train={components['total']:.4e}, "
                f"probe={validation_value:.4e}, "
                f"probe_miou={validation_miou:.4f}, "
                f"best={best_loss:.4e}@{best_step}"
            )
            decoder.train()
        del query, target, prediction, loss
        if (
            step >= int(config.minimum_steps)
            and step - last_improvement >= int(config.patience_steps)
        ):
            fabric.print(f">> early stop at step {step}; best step {best_step}")
            break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    module = getattr(decoder, "module", decoder)
    module.load_state_dict(best_state, strict=True)
    decoder.eval()

    full_prediction = _render_full_grid(
        decoder,
        latent_grid,
        full_query_gpu,
        outer_chunk_size=int(config.full_render_chunk_size),
    )
    grid_dim = int(model_config.dset.grid_dim)
    resolution = float(model_config.dset.resolution)
    prediction_grid = full_prediction.permute(0, 2, 1).reshape(
        -1,
        int(model_config.dset.n_channels),
        grid_dim,
        grid_dim,
        grid_dim,
    )
    predicted_atoms = get_atom_coords_batched(
        prediction_grid.clone(),
        fabric,
        rad=float(model_config.dset.ligand_radius),
        resolution=resolution,
        verbose=False,
        flexible=True,
    )[0]
    full_grid = {
        "reconstruction": reconstruction_metrics(
            full_prediction, full_target
        ),
        "atom_recovery": atom_recovery_metrics(
            predicted_atoms,
            batch_cpu["ligand"],
            grid_dim=grid_dim,
            resolution=resolution,
            match_distance=float(config.atom_match_distance),
        ),
    }
    threshold = float(config.occupancy_point_threshold)
    _atomic_sparse_occupancy_dump(
        output_dir / "reference_occupancy_points.npz",
        query_points=full_query,
        occupancy=full_target,
        threshold=threshold,
        grid_dim=grid_dim,
        resolution=resolution,
    )
    _atomic_sparse_occupancy_dump(
        output_dir / "gaussian_splat_occupancy_points.npz",
        query_points=full_query,
        occupancy=full_prediction,
        threshold=threshold,
        grid_dim=grid_dim,
        resolution=resolution,
    )
    with torch.no_grad():
        parameters = module.decode_parameters(latent_grid)
    domain_half_width = resolution * grid_dim / 2.0
    parameter_path = output_dir / "gaussian_parameters.npz"
    np.savez_compressed(
        parameter_path,
        centers=(
            parameters["centers"].detach().float().cpu().numpy()
            * domain_half_width
        ),
        scales=(
            parameters["scales"].detach().float().cpu().numpy()
            * domain_half_width
        ),
        opacities=parameters["opacities"].detach().float().cpu().numpy(),
    )
    fabric.save(
        output_dir / "gaussian_decoder_model.pt",
        {
            "decoder_state_dict": best_state,
            "config": model_config,
            "best_step": best_step,
            "best_validation_loss": best_loss,
        },
    )
    result = {
        "status": "complete",
        "experiment": "maximum same-target Gaussian decoder overfit",
        "checkpoint_encoder": str(checkpoint_path),
        "encoder_frozen": True,
        "latent": "deterministic posterior mean",
        "decoder_initialization": "fresh",
        "decoder_parameters": decoder_parameters,
        "gaussians_per_voxel": int(
            model_config.decoder.gaussians_per_voxel
        ),
        "dataset": str(config.dataset),
        "split": str(config.split),
        "sample_index": sample_index,
        "steps_completed": step,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "sampling_pool_sizes": {
            name: int(values.numel()) for name, values in pools.items()
        },
        "history": history,
        "full_grid": full_grid,
        "peak_gpu_memory_gib": (
            float(torch.cuda.max_memory_allocated(fabric.device) / 1024**3)
            if fabric.device.type == "cuda"
            else None
        ),
    }
    _atomic_json_dump(result, output_dir / "result.json")
    fabric.print(
        f">> full-grid Gaussian overfit: "
        f"mse={full_grid['reconstruction']['density_mse']:.4e}, "
        f"miou={full_grid['reconstruction']['miou']:.4f}, "
        f"predicted={full_grid['atom_recovery']['n_predicted_atoms']}, "
        f"matched={full_grid['atom_recovery']['matched_atoms']}"
    )
    fabric.print(f">> result written to {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
