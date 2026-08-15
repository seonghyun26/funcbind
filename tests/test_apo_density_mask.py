"""Checks for the apo (ligand-erased) density mask used by holo_density_fusion.

These run on CPU against the real 5MGL/7MU crop and the ligand atoms exported
by the holo run, so they verify the mask on the same data the GPU experiment
consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from funcbind.holo_density_fusion import (
    LIGAND_ELEMENTS,
    _apply_apo_mask,
    _ligand_occupancy_mask,
)
from funcbind.utils.constants import PADDING_INDEX


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLO_RUN = PROJECT_ROOT / "exps/density_fusion/holo_5mgl_target69_20260728"
CROP = Path(
    "/home1/irteam/VoxBind/voxbind/dataset/data/xray_crops_aligned_v5"
    "/test/000069.npy"
)
GRID_DIM = 64
RESOLUTION = 0.25

pytestmark = pytest.mark.skipif(
    not (HOLO_RUN / "reference_atoms.npz").is_file() or not CROP.is_file(),
    reason="requires the 5MGL holo run artifacts and the aligned density crop",
)


def _mask_config(**overrides):
    config = {
        "radius_scale": 1.25,
        "mask_floor": 0.5,
        "solvent_min_distance": 4.0,
        "min_solvent_voxels": 1000,
        "envelope_radius": 2.0,
        "leakage_sigma": 1.5,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _batch():
    """Rebuild the ligand/receptor entries the fusion script passes around."""
    atoms = np.load(HOLO_RUN / "reference_atoms.npz")
    coords = torch.from_numpy(atoms["coordinates"]).float()[None]
    channels = torch.from_numpy(atoms["channels"]).float()[None]
    radius = torch.ones_like(channels)

    # A receptor shell around the pocket: far enough from the ligand that it
    # must survive masking, close enough to constrain the solvent estimate.
    angles = torch.linspace(0, 2 * torch.pi, 40)[:-1]
    shell = torch.stack(
        [6.5 * torch.cos(angles), 6.5 * torch.sin(angles), torch.zeros_like(angles)],
        dim=-1,
    )[None]
    return {
        "ligand": {
            "coords": coords,
            "atoms_channel": channels,
            "radius": radius,
        },
        "receptor": {
            "coords": shell,
            "atoms_channel": torch.zeros(shell.shape[:2]),
        },
    }


def _density():
    return np.load(CROP).astype(np.float32)


def test_occupancy_mask_is_bounded_and_peaks_at_atoms():
    batch = _batch()
    mask = _ligand_occupancy_mask(
        batch,
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        radius_scale=1.0,
        device=torch.device("cpu"),
    )
    assert mask.shape == (GRID_DIM, GRID_DIM, GRID_DIM)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0

    coords = batch["ligand"]["coords"][0]
    index = torch.round(coords / RESOLUTION + (GRID_DIM - 1) / 2.0).long()
    at_atoms = mask[index[:, 0], index[:, 1], index[:, 2]]
    # Nearest voxel, so up to half a voxel diagonal (0.22 A) off centre.
    assert float(at_atoms.min()) > 0.95, "occupancy must saturate at atom centers"
    # Far corners of the 16 A box hold no ligand.
    assert float(mask[0, 0, 0]) < 1e-6


def test_padded_ligand_atoms_are_ignored():
    batch = _batch()
    n_real = batch["ligand"]["coords"].shape[1]
    reference = _ligand_occupancy_mask(
        batch,
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        radius_scale=1.0,
        device=torch.device("cpu"),
    )
    padded = {
        "ligand": {
            "coords": torch.cat(
                [batch["ligand"]["coords"], torch.zeros(1, 5, 3)], dim=1
            ),
            "atoms_channel": torch.cat(
                [
                    batch["ligand"]["atoms_channel"],
                    torch.full((1, 5), float(PADDING_INDEX)),
                ],
                dim=1,
            ),
            "radius": torch.cat(
                [batch["ligand"]["radius"], torch.full((1, 5), float(PADDING_INDEX))],
                dim=1,
            ),
        },
        "receptor": batch["receptor"],
    }
    assert padded["ligand"]["coords"].shape[1] == n_real + 5
    with_padding = _ligand_occupancy_mask(
        padded,
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        radius_scale=1.0,
        device=torch.device("cpu"),
    )
    assert torch.allclose(reference, with_padding)
    assert all(channel < len(LIGAND_ELEMENTS) for channel in range(4))


def test_apo_mask_removes_ligand_density_and_keeps_the_rest():
    density = _density()
    masked, mask, report = _apply_apo_mask(
        density,
        _batch(),
        _mask_config(),
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        device=torch.device("cpu"),
    )
    assert masked.shape == density.shape
    assert masked.dtype == np.float32

    # The holo map has strong ligand density; the masked map must not.
    assert report["ligand_atom_density_before"] > 1.5
    assert abs(report["ligand_atom_density_after"] - report["fill_value"]) < 0.05
    assert report["envelope_above_sigma_before"] > 0.1
    assert report["envelope_above_sigma_after"] < 0.01

    # Everything outside the mask is untouched, and the mask is local.
    assert report["unmasked_voxels_unchanged"]
    untouched = mask <= 0.0
    assert untouched.sum() == report["unmasked_voxels"] > 0
    assert np.array_equal(masked[untouched], density[untouched])
    assert report["voxels_touched_fraction"] < 0.15

    # The receptor shell at 6.5 A sits outside the ligand envelope.
    assert mask.sum() > 0
    assert report["fill_source"].startswith("median of")


def test_larger_radius_scale_erases_more():
    density = _density()
    fractions = []
    for radius_scale in (1.0, 1.5, 2.0):
        _, mask, report = _apply_apo_mask(
            density,
            _batch(),
            _mask_config(radius_scale=radius_scale),
            grid_dim=GRID_DIM,
            resolution=RESOLUTION,
            device=torch.device("cpu"),
        )
        fractions.append(report["voxels_touched_fraction"])
    assert fractions == sorted(fractions)
    assert fractions[0] < fractions[-1]
