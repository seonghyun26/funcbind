"""Ten-target apo-density FuncBind decoded by a jointly fitted Gaussian splatter.

This is the multi-target follow-up to the single-target apo control. Two things
change from that run, and both remove a way the earlier result could be
explained by memorization:

* One density adapter is fitted across all ten targets instead of one adapter
  per target, so it cannot reach any single reference code by memorizing a
  constant residual.
* The INR decoder is replaced by a Gaussian-splat decoder fitted jointly on the
  same ten targets' encoder codes, so the reported occupancies come from a
  decoder that had to serve ten codes rather than one.

The density fed to the encoder is apo: the measured 2Fo-Fc map with the
ligand's own density masked out by FuncBind's ligand occupancy field.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from funcbind.compare_decoders import _parameter_count
from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
from funcbind.dataset.field_maker import FieldMaker
from funcbind.distributions.distributions import UniformPlusNormal
from funcbind.holo_density_fusion import (
    LIGAND_ELEMENTS,
    DensityConditionAdapter,
    _apply_apo_mask,
    _atomic_json,
    _atoms_dump,
    _build_density_input,
    _density_cloud,
    _evaluate_code,
    _load_density_encoder,
    _resolve_path,
    _sample_codes,
    _seed_everything,
    _unwrap,
)
from funcbind.models.denoiser import FuncBind, add_noise_to_code
from funcbind.models.encoder import sample_posterior
from funcbind.train_fb import get_label
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_fb import (
    create_field_makers,
    load_unet,
    num_classes_funcbind,
)
from funcbind.utils.utils_nf import create_nf_decoder, create_nf_encoder
from funcbind.utils.utils_sampling import get_receptor_encoding, normalize_code


ARMS = (
    ("original_funcbind", "Original FuncBind"),
    ("apo_density_funcbind", "Apo-density adapter"),
)


def _load_models(config: DictConfig, fabric):
    """Frozen denoiser and ligand encoder, plus a fresh Gaussian decoder."""
    fb_path = _resolve_path(config.fb_pretrained_path) / "checkpoint.pth.tar"
    nf_path = _resolve_path(config.nf_pretrained_path) / "model.pt"
    for path in (fb_path, nf_path):
        if not path.is_file():
            raise FileNotFoundError(path)

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
        code_stats[key] = value.to(fabric.device) if torch.is_tensor(value) else value

    model = FuncBind(
        model_config,
        code_stats=code_stats,
        fabric=fabric,
        num_classes=num_classes_funcbind(model_config),
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

    encoder = create_nf_encoder(nf_config, fabric)
    encoder.load_state_dict(nf_checkpoint["enc_state_dict"], strict=True)
    del nf_checkpoint
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    encoder = fabric.setup_module(encoder)

    # The Gaussian decoder has no pretrained weights; it is fitted below.
    gaussian_config = copy.deepcopy(nf_config)
    gaussian_config.decoder = copy.deepcopy(config.gaussian_splat_decoder)
    decoder = create_nf_decoder(gaussian_config, fabric)
    decoder_module = _unwrap(decoder)
    decoder_module.set_code_stats(code_stats)

    return model, encoder, decoder, model_config, nf_config, gaussian_config, fb_path, nf_path


def _prepare_targets(
    config: DictConfig,
    model,
    ligand_encoder,
    model_config: DictConfig,
    nf_config: DictConfig,
    fabric,
) -> tuple[list[dict[str, Any]], int]:
    """Per-target codes, receptor conditioning and apo density features."""
    dataset = DatasetOmni(
        nf_config,
        split="test",
        sample_points=True,
        sample_full_grid=True,
        rebalance=False,
    )
    field_maker, field_maker_receptor = create_field_makers(
        model_config, nf_config, fabric
    )
    density_encoder, density_checkpoint = _load_density_encoder(config, fabric)
    crop_root = _resolve_path(config.density.crop_root)
    num_classes = num_classes_funcbind(model_config)

    targets = []
    for target_index in list(config.density.target_indices):
        target_index = int(target_index)
        if not 0 <= target_index < len(dataset):
            raise IndexError(f"target {target_index} outside test set {len(dataset)}")
        crop_path = crop_root / f"{target_index:06d}.npy"
        if not crop_path.is_file():
            raise FileNotFoundError(f"no aligned density crop for target {target_index}")

        batch = fabric.to_device(collate_fn([dataset[target_index]]))
        with torch.no_grad():
            voxels = field_maker.compute_voxel_grid(
                batch["ligand"], num_channels=len(LIGAND_ELEMENTS)
            )
            moments = ligand_encoder(voxels)
            latent_grid, _ = sample_posterior(
                moments, sample_posterior=False, save_log_var=False, deterministic=True
            )
            ligand_code = normalize_code(latent_grid, _unwrap(model).code_stats).detach()
            receptor_base = get_receptor_encoding(
                batch["receptor"],
                model,
                field_maker_receptor,
                fabric,
                rand_rots=None,
                n_chains=1,
            ).detach()
            occupancy = field_maker.compute_occupancies(
                batch["ligand"], num_channels=len(LIGAND_ELEMENTS)
            ).float().cpu()

        holo_density = np.load(crop_path).astype(np.float32)
        if holo_density.shape != (64, 64, 64):
            raise ValueError(f"target {target_index}: bad crop {holo_density.shape}")
        apo_density, apo_mask, mask_report = _apply_apo_mask(
            holo_density,
            batch,
            config.density.apo_mask,
            grid_dim=64,
            resolution=0.25,
            device=fabric.device,
        )
        density_input = _build_density_input(
            batch,
            apo_density,
            _resolve_path(config.density.voxbind_python_root),
            fabric,
        )
        with torch.no_grad(), fabric.autocast():
            tokens = density_encoder.forward_features(density_input)
            pooled = _unwrap(density_encoder)._pool_groups(tokens)
            density_feature = pooled.transpose(1, 2).reshape(1, 512, 8, 8, 8).detach()
        del tokens, pooled, density_input

        fabric.print(
            f">> target {target_index:3d}: apo mask "
            f"{mask_report['ligand_atom_density_before']:.2f} -> "
            f"{mask_report['ligand_atom_density_after']:.2f}, "
            f"envelope leakage {mask_report['envelope_above_sigma_after']:.4f}"
        )
        targets.append(
            {
                "index": target_index,
                "batch": batch,
                "ligand_code": ligand_code,
                "latent_grid": latent_grid.detach(),
                "receptor_base": receptor_base,
                "label": get_label(batch, model_config, num_classes=num_classes),
                "density_feature": density_feature,
                "occupancy": occupancy,
                "query_points": batch["ligand"]["xs"],
                "apo_density": apo_density,
                "apo_mask": apo_mask,
                "holo_density": holo_density,
                "mask_report": mask_report,
                "receptor_id": str(batch["receptor"]["id"][0]),
                "ligand_file_id": str(batch["ligand"]["id"][0]),
            }
        )

    del density_encoder
    torch.cuda.empty_cache()
    return targets, len(dataset), field_maker, str(density_checkpoint)


def _fit_joint_adapter(
    adapter,
    optimizer,
    targets: list[dict[str, Any]],
    model,
    config: DictConfig,
    fabric,
) -> tuple[dict[int, torch.Tensor], list[dict[str, float]]]:
    """One adapter over all targets: it cannot memorize a single reference code."""
    steps = int(config.adapter.steps)
    report_every = int(config.adapter.report_every)
    regularization = float(config.adapter.residual_regularization)
    history: list[dict[str, float]] = []
    adapter.train()
    model.eval()
    started = time.time()

    for step in range(1, steps + 1):
        target = targets[(step - 1) % len(targets)]
        optimizer.zero_grad(set_to_none=True)
        sigma = _unwrap(model).sigma_distribution.sample((1,)).to(fabric.device)
        noisy = add_noise_to_code(target["ligand_code"], sigma=sigma)
        with fabric.autocast():
            delta = adapter(target["density_feature"])
            prediction, logvar = model(
                ligand_encoding=noisy,
                sigma=sigma,
                receptor_encoding=target["receptor_base"] + delta,
                classes=target["label"],
                return_logvar=True,
                cfg_dropout=False,
            )
            sigma5 = sigma.reshape(-1, 1, 1, 1, 1)
            weight = (sigma5.square() + 1.0) / sigma5.square()
            squared_error = (prediction - target["ligand_code"]).square()
            denoise_loss = (weight / logvar.exp() * squared_error + logvar).mean()
            residual_loss = delta.float().square().mean()
            loss = denoise_loss + regularization * residual_loss
        fabric.backward(loss)
        fabric.clip_gradients(
            adapter, optimizer, max_norm=float(config.adapter.gradient_clip_norm)
        )
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == steps:
            record = {
                "step": step,
                "target_index": target["index"],
                "loss": float(loss.detach()),
                "denoise_loss": float(denoise_loss.detach()),
                "residual_rms": float(residual_loss.detach().sqrt()),
                "sigma": float(sigma.detach()),
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            fabric.print(
                f">> adapter {step:05d}/{steps} (target {target['index']:3d}): "
                f"loss={record['loss']:.5f}, residual_rms={record['residual_rms']:.4f}"
            )

    adapter.eval()
    fused = {}
    with torch.no_grad(), fabric.autocast():
        for target in targets:
            fused[target["index"]] = (
                target["receptor_base"] + adapter(target["density_feature"])
            ).detach()
    return fused, history


def _occupancy_pools(
    occupancy: torch.Tensor, config: DictConfig
) -> dict[str, torch.Tensor]:
    peak = occupancy.amax(dim=-1)[0]
    positive = peak >= float(config.positive_threshold)
    tail = (peak >= float(config.tail_threshold)) & ~positive
    return {
        "positive": torch.where(positive)[0],
        "tail": torch.where(tail)[0],
        "background": torch.where(peak < float(config.tail_threshold))[0],
    }


def _stratified_indices(
    pools: dict[str, torch.Tensor], total: int, config: DictConfig, generator
) -> tuple[torch.Tensor, dict[str, slice]]:
    counts = {
        "positive": int(round(total * float(config.positive_fraction))),
        "tail": int(round(total * float(config.tail_fraction))),
    }
    counts["background"] = total - counts["positive"] - counts["tail"]
    if min(counts.values()) <= 0:
        raise ValueError("every occupancy stratum must receive at least one point")
    pieces, slices, start = {}, {}, 0
    for name, count in counts.items():
        pool = pools[name]
        if pool.numel() == 0:
            raise ValueError(f"empty {name} stratum")
        positions = torch.randint(
            pool.numel(), (count,), generator=generator, dtype=torch.long
        )
        pieces[name] = pool[positions]
        slices[name] = slice(start, start + count)
        start += count
    return torch.cat([pieces[name] for name in counts]), slices


def _balanced_loss(prediction, target, slices, config):
    total = 0.0
    weights = {
        "positive": float(config.positive_loss_weight),
        "tail": float(config.tail_loss_weight),
        "background": float(config.background_loss_weight),
    }
    for name, weight in weights.items():
        piece_prediction = prediction[:, slices[name]]
        piece_target = target[:, slices[name]]
        point_weight = 1.0 + float(config.target_value_weight) * piece_target
        total = total + weight * (
            point_weight * (piece_prediction - piece_target).square()
        ).mean()
    return total


def _fit_joint_gaussian_decoder(
    decoder, targets: list[dict[str, Any]], config: DictConfig, fabric
) -> list[dict[str, float]]:
    """One Gaussian-splat decoder serving all ten targets' encoder codes."""
    fit = config.decoder_fit
    optimizer = torch.optim.Adam(decoder.parameters(), lr=float(fit.lr))
    decoder, optimizer = fabric.setup(decoder, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(fit.steps), eta_min=float(fit.min_lr)
    )

    pools = [_occupancy_pools(target["occupancy"], fit) for target in targets]
    validation_generator = torch.Generator().manual_seed(int(config.seed) + 100_000)
    validation = []
    for target, pool in zip(targets, pools):
        indices, slices = _stratified_indices(
            pool, int(fit.validation_points), fit, validation_generator
        )
        validation.append(
            {
                "query": target["query_points"][:, indices].to(fabric.device),
                "target": target["occupancy"][:, indices].to(fabric.device),
                "slices": slices,
                "latent": target["latent_grid"],
            }
        )

    training_generator = torch.Generator().manual_seed(int(config.seed) + 1)
    history: list[dict[str, float]] = []
    best_loss, best_state, best_step = float("inf"), None, 0
    started = time.time()

    for step in range(1, int(fit.steps) + 1):
        position = (step - 1) % len(targets)
        target, pool = targets[position], pools[position]
        indices, slices = _stratified_indices(
            pool, int(fit.batch_points), fit, training_generator
        )
        query = target["query_points"][:, indices].to(fabric.device)
        reference = target["occupancy"][:, indices].to(fabric.device)
        optimizer.zero_grad()
        prediction = decoder(query, target["latent_grid"])
        loss = _balanced_loss(prediction, reference, slices, fit)
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), float(fit.gradient_clip_norm)
        )
        optimizer.step()
        scheduler.step()

        if step % int(fit.report_every) == 0 or step == 1:
            decoder.eval()
            with torch.no_grad():
                # Validate on every target, so the score reflects all ten.
                losses = [
                    float(
                        _balanced_loss(
                            decoder(item["query"], item["latent"]),
                            item["target"],
                            item["slices"],
                            fit,
                        )
                    )
                    for item in validation
                ]
            decoder.train()
            mean_loss = float(np.mean(losses))
            if mean_loss < best_loss - float(fit.min_delta):
                best_loss, best_step = mean_loss, step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in _unwrap(decoder).state_dict().items()
                }
            record = {
                "step": step,
                "training_loss": float(loss.detach()),
                "validation_loss": mean_loss,
                "validation_worst": float(np.max(losses)),
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            fabric.print(
                f">> decoder {step:05d}/{int(fit.steps)}: "
                f"train={record['training_loss']:.3e}, "
                f"val={mean_loss:.3e} (worst {record['validation_worst']:.3e})"
            )

    if best_state is not None:
        _unwrap(decoder).load_state_dict(best_state)
        fabric.print(f">> restored decoder from step {best_step} (val {best_loss:.3e})")
    decoder.eval()
    return history, best_step, best_loss


@hydra.main(
    config_path="configs", config_name="apo_gaussian_multi_target", version_base=None
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
        gaussian_config,
        fb_checkpoint,
        nf_checkpoint,
    ) = _load_models(config, fabric)
    OmegaConf.save(gaussian_config, output_dir / "effective_nf_config.yaml")

    targets, test_size, field_maker, density_checkpoint = _prepare_targets(
        config, model, ligand_encoder, model_config, nf_config, fabric
    )
    del ligand_encoder
    torch.cuda.empty_cache()

    adapter = DensityConditionAdapter()
    adapter_optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config.adapter.learning_rate),
        weight_decay=float(config.adapter.weight_decay),
    )
    adapter, adapter_optimizer = fabric.setup(adapter, adapter_optimizer)
    fused, adapter_history = _fit_joint_adapter(
        adapter, adapter_optimizer, targets, model, config, fabric
    )
    torch.save(
        {
            "adapter_state_dict": _unwrap(adapter).state_dict(),
            "history": adapter_history,
            "target_indices": [target["index"] for target in targets],
        },
        output_dir / "density_adapter.pt",
    )
    _atomic_json(output_dir / "adapter_history.json", adapter_history)

    decoder_history, best_step, best_loss = _fit_joint_gaussian_decoder(
        decoder, targets, config, fabric
    )
    decoder_module = _unwrap(decoder)
    torch.save(
        {"decoder_state_dict": decoder_module.state_dict(), "history": decoder_history},
        output_dir / "gaussian_decoder.pt",
    )
    _atomic_json(output_dir / "decoder_history.json", decoder_history)

    n_chains = int(config.sampling.n_chains)
    initial_distribution = UniformPlusNormal(
        _unwrap(model).code_stats,
        model_config,
        device=fabric.device,
        dtype=torch.float32,
    )

    per_target = []
    for target in targets:
        target_dir = output_dir / f"target_{target['index']:03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        label_n = (
            target["label"].repeat(n_chains, 1)
            if target["label"] is not None
            else None
        )
        conditions = {
            "original_funcbind": target["receptor_base"],
            "apo_density_funcbind": fused[target["index"]],
        }
        y_init = initial_distribution.sample()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        codes, latent_mse, best_index = {}, {}, {}
        target_code = target["ligand_code"].float().cpu()
        for key, label in ARMS:
            fabric.print(f">> target {target['index']:3d} sampling: {label}")
            sampled = _sample_codes(
                model,
                conditions[key].repeat(n_chains, 1, 1, 1, 1),
                label_n,
                y_init,
                model_config,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
            )
            codes[key] = sampled
            latent_mse[key] = (
                (sampled - target_code).square().flatten(1).mean(1)
            )
            best_index[key] = int(latent_mse[key].argmin())

        _atoms_dump(
            target_dir / "reference_atoms.npz",
            {
                "coords": target["batch"]["ligand"]["coords"][
                    :, target["batch"]["ligand"]["atoms_channel"][0]
                    < len(LIGAND_ELEMENTS)
                ],
                "atoms_channel": target["batch"]["ligand"]["atoms_channel"][
                    :, target["batch"]["ligand"]["atoms_channel"][0]
                    < len(LIGAND_ELEMENTS)
                ],
            },
        )
        np.savez_compressed(
            target_dir / "apo_mask.npz",
            mask=target["apo_mask"].astype(np.float16),
            holo_density=target["holo_density"].astype(np.float16),
            apo_density=target["apo_density"].astype(np.float16),
            resolution=np.asarray(0.25, dtype=np.float32),
        )
        _density_cloud(
            target_dir / "density_points.npz",
            target["apo_density"],
            float(config.density.density_point_threshold),
        )

        results = {
            key: _evaluate_code(
                key,
                codes[key][best_index[key] : best_index[key] + 1],
                decoder_module,
                target["batch"],
                target["occupancy"],
                target["query_points"],
                target_dir,
                config,
                gaussian_config,
                fabric,
            )
            for key, _ in ARMS
        }
        per_target.append(
            {
                "target_index": target["index"],
                "receptor_id": target["receptor_id"],
                "ligand_file_id": target["ligand_file_id"],
                "apo_mask": target["mask_report"],
                "latent_mse": {key: latent_mse[key].tolist() for key, _ in ARMS},
                "best_index": best_index,
                "results": results,
            }
        )
        _atomic_json(output_dir / "per_target_partial.json", per_target)
        del codes
        torch.cuda.empty_cache()

    def _aggregate(metric_path: tuple[str, ...]) -> dict[str, dict[str, float]]:
        summary = {}
        for key, _ in ARMS:
            values = []
            for entry in per_target:
                node = entry["results"][key]
                for step in metric_path:
                    node = node[step]
                values.append(float(node))
            summary[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        return summary

    paired = {}
    for key, _ in ARMS:
        if key == "original_funcbind":
            continue
        deltas = [
            entry["latent_mse"][key][entry["best_index"][key]]
            - entry["latent_mse"]["original_funcbind"][
                entry["best_index"]["original_funcbind"]
            ]
            for entry in per_target
        ]
        paired[key] = {
            "latent_mse_delta": deltas,
            "targets_improved": int(sum(1 for value in deltas if value < 0)),
            "n_targets": len(deltas),
        }

    report = {
        "status": "complete",
        "experiment": (
            "ten-target apo-density FuncBind with a jointly fitted "
            "Gaussian-splat decoder"
        ),
        "targets": [target["index"] for target in targets],
        "test_set_size": test_size,
        "checkpoints": {
            "funcbind": str(fb_checkpoint),
            "neural_field_encoder": str(nf_checkpoint),
            "density_encoder": density_checkpoint,
        },
        "decoder": {
            "type": "gaussian_splat",
            "fitted_jointly_on": [target["index"] for target in targets],
            "trainable_parameters": _parameter_count(decoder_module),
            "steps": int(config.decoder_fit.steps),
            "best_step": best_step,
            "best_validation_loss": best_loss,
            "note": (
                "no pretrained Gaussian decoder exists; it is fitted here on the "
                "frozen encoder's codes for these ten targets"
            ),
        },
        "adapter": {
            "trainable_parameters": sum(
                p.numel() for p in _unwrap(adapter).parameters()
            ),
            "steps": int(config.adapter.steps),
            "fitted_jointly_on": [target["index"] for target in targets],
            "final_training_record": adapter_history[-1],
        },
        "arms": [{"key": key, "label": label} for key, label in ARMS],
        "sampling": {
            "n_chains": n_chains,
            "paired_initial_latents": True,
            "paired_sampler_rng": True,
            "sampler_steps": int(model_config.sampler.N),
            "selection": "oracle minimum latent MSE per target and arm",
        },
        "aggregate": {
            "density_mse": _aggregate(("reconstruction", "density_mse")),
            "miou": _aggregate(("reconstruction", "miou")),
            "atom_f1": _aggregate(("atom_recovery", "atom_f1")),
            "atom_precision": _aggregate(("atom_recovery", "atom_precision")),
            "atom_recall": _aggregate(("atom_recovery", "atom_recall")),
        },
        "paired": paired,
        "per_target": per_target,
        "limitations": [
            "The apo map is a masked holo map, not a measured apo structure: "
            "solvent reorganization and side-chain relaxation on ligand release "
            "are not modeled, and receptor density inside the ligand envelope is "
            "erased with the ligand.",
            "The Gaussian decoder is fitted on these same ten targets' codes, so "
            "decoded quality is in-sample and is not a held-out decoder result.",
            "Chain selection per target and arm is an oracle on latent MSE.",
            "Ten targets is a small sample; per-target pairing is reported so the "
            "spread is visible rather than hidden in a mean.",
        ],
    }
    _atomic_json(output_dir / "result.json", report)
    os.replace(output_dir / "per_target_partial.json", output_dir / "per_target.json")
    fabric.print(f">> complete: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
