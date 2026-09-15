#!/usr/bin/env python
"""Validate one real MCP/X-ray target through the CDG v2 conditioning path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=REPO / "funcbind/dataset/data/mcpp_dataset"
    )
    parser.add_argument("--density-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--config", default="train_fb_mcpp_holo_density_default")
    parser.add_argument(
        "--voxbind-root",
        type=Path,
        default=Path(os.environ.get("VOXBIND_PYTHON_ROOT", REPO.parent)),
    )
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def check(name: str, good: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if good else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not good:
        raise RuntimeError(name)


def main() -> None:
    args = parse_args()
    os.environ["FUNCBIND_ROOT"] = str(REPO)
    os.environ["VOXBIND_PYTHON_ROOT"] = str(args.voxbind_root.resolve())
    if args.encoder_checkpoint:
        os.environ["FUNCBIND_DENSITY_ENCODER"] = str(args.encoder_checkpoint.resolve())

    with initialize_config_dir(
        config_dir=str(REPO / "funcbind/configs"), version_base=None
    ):
        cfg = compose(config_name=args.config, overrides=["wandb=false", "dset.data_aug=false"])
    config = OmegaConf.to_container(cfg, resolve=True)
    config["dset"]["data_dir"] = str(args.data_root.resolve().parent)
    config["dset"]["use_single_dataset"] = "mcpp"
    config["dset"]["mcpp_holo_density_dir"] = str(args.density_root.resolve())

    from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
    from funcbind.dataset.mcpp_holo_density import mcpp_target_id
    from funcbind.models.density_condition import DensityCondition, build_density_input, make_density_voxelizer

    dataset = DatasetOmni(
        config, split=args.split, sample_points=False, sample_full_grid=False, rebalance=False
    )
    index = next(
        (i for i, sample in enumerate(dataset.data) if mcpp_target_id(sample[2]) == args.target.lower()),
        None,
    )
    check("target exists in the selected split", index is not None, args.target)
    batch = collate_fn([dataset[index]])
    density = batch["density"]
    available = batch["density_available"]
    check("real X-ray density is available", bool(available.all()), str(available.tolist()))
    check("density crop has training shape", tuple(density.shape) == (1, 64, 64, 64), str(tuple(density.shape)))
    check("density crop is finite", bool(torch.isfinite(density).all()))
    check("density crop is non-constant", float(density.std()) > 0, f"std={density.std().item():.5f}")

    voxelizer = make_density_voxelizer("cpu", args.voxbind_root, backend="torch")
    density_input = build_density_input(
        batch["receptor"], density, voxelizer, args.voxbind_root
    )
    groups = list(config["denoiser"]["density"]["channel_groups"])
    check("CDG v2 input has 13 channels", density_input.shape[1] == sum(groups) == 13, str(groups))
    check("CDG v2 input is finite", bool(torch.isfinite(density_input).all()))

    if args.encoder_checkpoint:
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)
        density_cfg = dict(config["denoiser"]["density"])
        density_cfg["pretrained_path"] = str(args.encoder_checkpoint.resolve())
        condition = DensityCondition(
            density_cfg=density_cfg,
            code_dim=128,
            code_grid_dim=16,
            voxbind_root=args.voxbind_root,
            hidden=int(density_cfg.get("proj_hidden", 192)),
            freeze=True,
            amp=bool(density_cfg.get("encoder_amp", True)),
            latent_extent=32.0,
            fusion="default",
        ).to(device)
        condition.train()
        condition_tensor = torch.randn(1, 128, 16, 16, 16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            delta = condition(density_input.to(device), cond=condition_tensor)
        check("CDG v2 checkpoint loads strictly", sum(p.numel() for p in condition.encoder.parameters()) > 0)
        check("zero-init gives an exact no-op", int(torch.count_nonzero(delta)) == 0)
        target = torch.randn_like(delta)
        (delta.float() - target).square().mean().backward()
        gradient = condition.proj[-1].weight.grad
        check(
            "density projection receives a gradient",
            gradient is not None and bool(torch.count_nonzero(gradient)),
            f"max={gradient.abs().max().item():.3e}",
        )
        check("CDG v2 encoder remains frozen", not any(p.requires_grad for p in condition.encoder.parameters()))

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
