"""Render the holo -> apo density mask without touching a GPU.

Uses the ligand atoms exported by the holo run, so it shows exactly what
`holo_density_fusion.py apo_mask.enabled=true` will feed the density encoder.
The bulk-solvent fill is estimated from ligand distance only here; the real run
also excludes receptor neighbourhoods, which moves the fill by a few hundredths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from funcbind.holo_density_fusion import _apply_apo_mask


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
FIGURE_DIR = NOTEBOOK_DIR / "figures"
GRID_DIM = 64
RESOLUTION = 0.25


def _batch(atoms_path: Path) -> dict:
    atoms = np.load(atoms_path)
    coords = torch.from_numpy(atoms["coordinates"]).float()[None]
    channels = torch.from_numpy(atoms["channels"]).float()[None]
    return {
        "ligand": {
            "coords": coords,
            "atoms_channel": channels,
            "radius": torch.ones_like(channels),
        },
        # No receptor coordinates outside the fusion run; an empty set only
        # widens the solvent pool, which is the conservative direction.
        "receptor": {
            "coords": torch.zeros(1, 0, 3),
            "atoms_channel": torch.zeros(1, 0),
        },
    }


def render(crop_path: Path, atoms_path: Path, radius_scale: float) -> Path:
    density = np.load(crop_path).astype(np.float32)
    config = OmegaConf.create(
        {
            "radius_scale": radius_scale,
            "mask_floor": 0.5,
            "solvent_min_distance": 4.0,
            "min_solvent_voxels": 1000,
            "envelope_radius": 2.0,
            "leakage_sigma": 1.5,
        }
    )
    masked, mask, report = _apply_apo_mask(
        density,
        _batch(atoms_path),
        config,
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        device=torch.device("cpu"),
    )
    print(json.dumps(report, indent=2))

    mid = GRID_DIM // 2
    panels = (
        ("Holo 2Fo-Fc crop", density, "coolwarm"),
        ("Ligand occupancy mask", mask, "magma"),
        ("Apo (ligand erased)", masked, "coolwarm"),
    )
    vmax = float(np.quantile(np.abs(density), 0.995))
    figure, axes = plt.subplots(3, 3, figsize=(11.5, 11), constrained_layout=True)
    for row, (axis_name, take) in enumerate(
        (("x = 0 Å", lambda v: v[mid]), ("y = 0 Å", lambda v: v[:, mid]),
         ("z = 0 Å", lambda v: v[:, :, mid]))
    ):
        for column, (title, volume, cmap) in enumerate(panels):
            axis = axes[row, column]
            limits = dict(vmin=0, vmax=1) if cmap == "magma" else dict(
                vmin=-vmax, vmax=vmax
            )
            shown = axis.imshow(
                take(volume).T,
                origin="lower",
                cmap=cmap,
                extent=(-8, 8, -8, 8),
                **limits,
            )
            axis.set(xlabel="Å", ylabel="Å")
            axis.set_title(f"{title}\n{axis_name}", fontsize=10)
            figure.colorbar(shown, ax=axis, shrink=0.78)
    figure.suptitle(
        "Erasing the ligand from the 5MGL holo map "
        f"(occupancy radius x{radius_scale})",
        fontsize=14,
        fontweight="bold",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "apo_density_mask_preview.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop",
        type=Path,
        default=Path(
            "/home1/irteam/VoxBind/voxbind/dataset/data/xray_crops_aligned_v5"
            "/test/000069.npy"
        ),
    )
    parser.add_argument(
        "--atoms",
        type=Path,
        default=PROJECT_ROOT
        / "exps/density_fusion/holo_5mgl_target69_20260728/reference_atoms.npz",
    )
    parser.add_argument("--radius-scale", type=float, default=1.25)
    arguments = parser.parse_args()
    print(render(arguments.crop, arguments.atoms, arguments.radius_scale))
