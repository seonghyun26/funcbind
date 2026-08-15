"""Evaluate the complete pretrained FuncBind INR reconstruction pipeline."""

from __future__ import annotations

import json
import os
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
    atom_recovery_metrics,
    reconstruction_metrics,
)
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.dataset.field_maker import FieldMaker
from funcbind.models.decoder import get_atom_coords_batched, get_code_spatial
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


@torch.no_grad()
def _render_original_inr(
    encoder,
    decoder,
    voxels: torch.Tensor,
    query_points: torch.Tensor,
    *,
    posterior_mode: str,
    query_chunk_size: int,
    fabric,
) -> torch.Tensor:
    if posterior_mode not in {"sample", "mean"}:
        raise ValueError("posterior_mode must be 'sample' or 'mean'")
    encoder.eval()
    decoder.eval()
    with fabric.autocast():
        moments = encoder(voxels)
    predictions = []
    for query_chunk in query_points.split(query_chunk_size, dim=1):
        with fabric.autocast():
            spatial_moments = get_code_spatial(query_chunk, moments)
            latent, _ = sample_posterior(
                spatial_moments,
                sample_posterior=posterior_mode == "sample",
                save_log_var=False,
                deterministic=posterior_mode == "mean",
            )
            predictions.append(decoder(query_chunk, latent).float().cpu())
        del spatial_moments, latent
    return torch.cat(predictions, dim=1)


@hydra.main(
    config_path="configs",
    config_name="evaluate_pretrained_inr",
    version_base=None,
)
def main(config: DictConfig) -> None:
    package_root = Path(__file__).resolve().parent
    checkpoint_path = _resolve_path(
        str(config.pretrained_checkpoint), relative_to=package_root.parent
    )
    data_dir = _resolve_path(
        str(config.data_dir), relative_to=package_root
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(str(config.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model_config.decoder.type = "inr"
    model_config.decoder.compile = False
    OmegaConf.save(model_config, output_dir / "model_config.yaml")
    OmegaConf.save(config, output_dir / "evaluation_config.yaml")

    encoder = create_nf_encoder(model_config, fabric)
    decoder = create_nf_decoder(model_config, fabric)
    encoder.load_state_dict(checkpoint["enc_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["dec_state_dict"], strict=True)
    del checkpoint
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    encoder = fabric.setup_module(encoder)
    decoder = fabric.setup_module(decoder)

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
    _seed(int(config.seed))
    batch = fabric.to_device(collate_fn([dataset[sample_index]]))
    field_maker = FieldMaker(model_config).to(fabric.device)
    query_points = batch["ligand"]["xs"]
    voxels = field_maker.compute_voxel_grid(
        batch["ligand"], num_channels=int(model_config.dset.n_channels)
    )
    target = field_maker.compute_occupancies(
        batch["ligand"], num_channels=int(model_config.dset.n_channels)
    ).float().cpu()
    grid_dim = int(model_config.dset.grid_dim)
    resolution = float(model_config.dset.resolution)
    threshold = float(config.occupancy_point_threshold)
    _atomic_sparse_occupancy_dump(
        output_dir / "reference_occupancy_points.npz",
        query_points=query_points,
        occupancy=target,
        threshold=threshold,
        grid_dim=grid_dim,
        resolution=resolution,
    )

    results = {}
    for mode_index, posterior_mode in enumerate(config.posterior_modes):
        posterior_mode = str(posterior_mode)
        _seed(int(config.seed) + mode_index)
        prediction = _render_original_inr(
            encoder,
            decoder,
            voxels,
            query_points,
            posterior_mode=posterior_mode,
            query_chunk_size=int(config.evaluation_query_chunk_size),
            fabric=fabric,
        )
        prediction_grid = prediction.permute(0, 2, 1).reshape(
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
        result = {
            "posterior_mode": posterior_mode,
            "reconstruction": reconstruction_metrics(prediction, target),
            "atom_recovery": atom_recovery_metrics(
                predicted_atoms,
                batch["ligand"],
                grid_dim=grid_dim,
                resolution=resolution,
                match_distance=float(config.atom_match_distance),
            ),
        }
        results[posterior_mode] = result
        _atomic_sparse_occupancy_dump(
            output_dir
            / f"original_inr_{posterior_mode}_occupancy_points.npz",
            query_points=query_points,
            occupancy=prediction,
            threshold=threshold,
            grid_dim=grid_dim,
            resolution=resolution,
        )
        _atomic_json_dump(
            result, output_dir / f"original_inr_{posterior_mode}.json"
        )
        fabric.print(
            f">> original INR ({posterior_mode}): "
            f"mse={result['reconstruction']['density_mse']:.4e}, "
            f"miou={result['reconstruction']['miou']:.4f}, "
            f"predicted_atoms={result['atom_recovery']['n_predicted_atoms']}"
        )
        del prediction, prediction_grid, predicted_atoms
        if fabric.device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "status": "complete",
        "pipeline": "original pretrained FuncBind neural field",
        "checkpoint": str(checkpoint_path),
        "checkpoint_components": ["enc_state_dict", "dec_state_dict"],
        "dataset": str(config.dataset),
        "split": str(config.split),
        "sample_index": sample_index,
        "target_id": _jsonable(batch["ligand"].get("id")),
        "results": results,
    }
    _atomic_json_dump(report, output_dir / "result.json")
    fabric.print(f">> original pipeline result written to {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
