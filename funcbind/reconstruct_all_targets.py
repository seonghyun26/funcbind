"""Run a decoder reconstruction benchmark over an entire dataset split.

The single-target comparison in :mod:`funcbind.compare_decoders` is useful for
quick controlled checks.  This runner applies one decoder variant to every
requested target, writes one result per target, and continuously updates a
compact aggregate.  Separate processes can therefore run the INR and Gaussian
variants on different GPUs without sharing mutable output files.
"""

from __future__ import annotations

import gc
import json
import math
import random
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
    _train_variant,
)
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_nf import create_nf_encoder


METRIC_PATHS = {
    "sample_final_density_mse": ("final", "density_mse"),
    "sample_final_density_mae": ("final", "density_mae"),
    "sample_final_miou": ("final", "miou"),
    "mean_step_seconds": ("timing", "mean_step_seconds"),
    "peak_gpu_memory_gib": ("timing", "peak_gpu_memory_gib"),
    "full_density_mse": ("full_grid", "reconstruction", "density_mse"),
    "full_density_mae": ("full_grid", "reconstruction", "density_mae"),
    "full_miou": ("full_grid", "reconstruction", "miou"),
    "atom_precision": ("full_grid", "atom_recovery", "atom_precision"),
    "atom_recall": ("full_grid", "atom_recovery", "atom_recall"),
    "atom_f1": ("full_grid", "atom_recovery", "atom_f1"),
    "coordinate_rmsd": (
        "full_grid",
        "atom_recovery",
        "coordinate_rmsd",
    ),
    "element_accuracy": (
        "full_grid",
        "atom_recovery",
        "element_accuracy",
    ),
    "element_aware_precision": (
        "full_grid",
        "atom_recovery",
        "element_aware_precision",
    ),
    "element_aware_recall": (
        "full_grid",
        "atom_recovery",
        "element_aware_recall",
    ),
}


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _metric_stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=np.float64
    )
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    variant: str,
    dataset_size: int,
    requested_indices: list[int],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-safe aggregate from completed per-target results."""
    metrics: dict[str, Any] = {}
    for name, path in METRIC_PATHS.items():
        values = []
        for result in results:
            value = _nested_value(result, path)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        metrics[name] = _metric_stats(values)

    atom_totals = {}
    for name in (
        "n_target_atoms",
        "n_predicted_atoms",
        "matched_atoms",
    ):
        values = [
            _nested_value(result, ("full_grid", "atom_recovery", name))
            for result in results
        ]
        atom_totals[name] = int(
            sum(value for value in values if isinstance(value, (int, float)))
        )

    compact_targets = []
    for result in sorted(results, key=lambda item: int(item["sample_index"])):
        compact_targets.append(
            {
                "sample_index": int(result["sample_index"]),
                "target_id": result.get("target_id"),
                **{
                    name: _nested_value(result, path)
                    for name, path in METRIC_PATHS.items()
                },
            }
        )

    return {
        "status": (
            "complete"
            if len(results) + len(failures) == len(requested_indices)
            else "running"
        ),
        "variant": variant,
        "dataset_size": int(dataset_size),
        "requested_count": len(requested_indices),
        "completed_count": len(results),
        "failure_count": len(failures),
        "completed_indices": sorted(
            int(result["sample_index"]) for result in results
        ),
        "failed_indices": sorted(
            int(failure["sample_index"]) for failure in failures
        ),
        "metrics": metrics,
        "atom_totals": atom_totals,
        "targets": compact_targets,
        "failures": failures,
    }


def _seed_data_sampling(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_id(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_id(item) for item in value]
    return str(value)


def _resolve_data_dir(config: DictConfig) -> None:
    data_dir = Path(str(config.dset.data_dir)).expanduser()
    if data_dir.is_absolute():
        return
    candidates = [
        Path.cwd() / data_dir,
        Path(__file__).resolve().parent / data_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            config.dset.data_dir = str(candidate.resolve())
            return
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve dset.data_dir; searched {searched}")


def _load_completed_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return result if result.get("status") == "complete" else None


@hydra.main(
    config_path="configs",
    config_name="reconstruct_all_targets",
    version_base=None,
)
def main(config: DictConfig) -> None:
    variant = str(config.variant)
    if variant not in {"inr", "gaussian_splat"}:
        raise ValueError("variant must be 'inr' or 'gaussian_splat'")
    if config.reg_weight != 0:
        raise ValueError("The controlled reconstruction requires reg_weight=0")

    _resolve_data_dir(config)
    output_dir = Path(str(config.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    fabric = setup_fabric(config)
    sampled_dataset = DatasetOmni(
        config,
        split=str(config.split),
        sample_points=True,
        sample_full_grid=False,
        rebalance=False,
    )
    full_grid_dataset = (
        DatasetOmni(
            config,
            split=str(config.split),
            sample_points=True,
            sample_full_grid=True,
            rebalance=False,
        )
        if bool(config.full_grid_metrics)
        else None
    )
    dataset_size = len(sampled_dataset)
    end_index = (
        dataset_size if config.end_index is None else int(config.end_index)
    )
    start_index = int(config.start_index)
    if not 0 <= start_index <= end_index <= dataset_size:
        raise ValueError(
            f"Requested [{start_index}, {end_index}) for dataset size "
            f"{dataset_size}"
        )
    requested_indices = list(range(start_index, end_index))

    base_config = _decoder_config(config, "inr")
    fabric.seed_everything(int(config.seed))
    initial_encoder = create_nf_encoder(base_config, fabric)
    initial_encoder_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_encoder.state_dict().items()
    }
    del initial_encoder
    gc.collect()
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    result_name = f"{variant}.json"

    for position, sample_index in enumerate(requested_indices, start=1):
        target_dir = output_dir / f"target_{sample_index:03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        result_path = target_dir / result_name
        completed = (
            _load_completed_result(result_path)
            if bool(config.resume)
            else None
        )
        if completed is not None:
            results.append(completed)
            fabric.print(
                f">> [{position}/{len(requested_indices)}] target "
                f"{sample_index}: already complete"
            )
            continue

        fabric.print(
            f">> [{position}/{len(requested_indices)}] target "
            f"{sample_index}: {variant}"
        )
        try:
            _seed_data_sampling(int(config.seed) + sample_index)
            batch = fabric.to_device(
                collate_fn([sampled_dataset[sample_index]])
            )
            full_grid_batch = (
                fabric.to_device(
                    collate_fn([full_grid_dataset[sample_index]])
                )
                if full_grid_dataset is not None
                else None
            )
            target_id = _jsonable_id(batch["ligand"].get("id"))
            result = _train_variant(
                variant,
                config,
                fabric,
                batch,
                initial_encoder_state,
                target_dir,
                full_grid_batch,
            )
            result["sample_index"] = sample_index
            result["target_id"] = target_id
            _atomic_json_dump(result, result_path)
            results.append(result)
            del batch, full_grid_batch, result
        except Exception as exc:
            failure = {
                "sample_index": sample_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _atomic_json_dump(failure, target_dir / "error.json")
            fabric.print(
                f">> target {sample_index} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if bool(config.fail_fast):
                raise
        finally:
            gc.collect()
            if fabric.device.type == "cuda":
                torch.cuda.empty_cache()

        summary = aggregate_results(
            results,
            variant=variant,
            dataset_size=dataset_size,
            requested_indices=requested_indices,
            failures=failures,
        )
        _atomic_json_dump(summary, output_dir / "summary.json")

    summary = aggregate_results(
        results,
        variant=variant,
        dataset_size=dataset_size,
        requested_indices=requested_indices,
        failures=failures,
    )
    _atomic_json_dump(summary, output_dir / "summary.json")
    fabric.print(
        f">> {variant} all-target reconstruction finished: "
        f"{len(results)} complete, {len(failures)} failed; "
        f"summary at {output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
