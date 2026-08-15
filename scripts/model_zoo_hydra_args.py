#!/usr/bin/env python3
"""Validate a VoxBind model-zoo encoder and emit downstream Hydra overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    config = OmegaConf.load(args.config)
    model = config.model
    required = (
        "patch_size",
        "n_in_channels",
        "dim",
        "depth",
        "heads",
        "mlp_ratio",
        "dropout",
        "patch_embed_mode",
        "channel_groups",
    )
    missing = [key for key in required if model.get(key) is None]
    if missing:
        raise ValueError(
            f"{args.config} is missing model fields: {', '.join(missing)}"
        )

    input_channels = int(model.n_in_channels)
    channel_groups = [int(value) for value in model.channel_groups]
    if input_channels != 13:
        raise ValueError(
            f"{args.config.parent.name} has n_in_channels={input_channels}. "
            "Frozen holo-density VoxBind requires the full 13-channel "
            "[7 ligand, 4 pocket, density, gradmag] encoder; coordinate-only "
            "control encoders are not compatible."
        )
    if sum(channel_groups) != input_channels:
        raise ValueError(
            f"channel_groups={channel_groups} sum to {sum(channel_groups)}, "
            f"not n_in_channels={input_channels}"
        )

    values = {
        "model.density_vit.patch": int(model.patch_size),
        "model.density_vit.dim": int(model.dim),
        "model.density_vit.depth": int(model.depth),
        "model.density_vit.heads": int(model.heads),
        "model.density_vit.mlp_ratio": int(model.mlp_ratio),
        "model.density_vit.dropout": float(model.dropout),
        "model.density_vit.n_in_channels": input_channels,
        "model.density_vit.patch_embed_mode": str(model.patch_embed_mode),
        "model.density_vit.channel_groups": channel_groups,
    }
    for key, value in values.items():
        if isinstance(value, list):
            rendered = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, float):
            rendered = repr(value)
        else:
            rendered = str(value)
        print(f"{key}={rendered}")


if __name__ == "__main__":
    main()
