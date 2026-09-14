"""Launch VoxBind DDP with two numerically equivalent throughput fixes.

The upstream training entrypoint already contains a fused ``foreach`` EMA
implementation, but constructs it with the slow default.  This wrapper enables
that existing path.  It also uses two PyUUL chunks for a per-rank batch of 32
instead of four.  VoxBind adds fixed -25/+25 Angstrom sentinel coordinates
before every PyUUL call, so chunk grouping cannot change the centered 64^3 crop;
it only reduces repeated CUDA scalar synchronizations and allocations.

The custom AdamW optimizer remains untouched.  Its foreach implementation
omits the single-tensor path's zero-denominator guard and is therefore not safe
to enable for this run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


VOXBIND_ROOT = Path(os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind")) / "voxbind"
if str(VOXBIND_ROOT) not in sys.path:
    sys.path.insert(0, str(VOXBIND_ROOT))

import train_ddp as upstream
from voxbind.models.ema import ModelEma as UpstreamModelEma
from voxbind.voxelizer import Voxelizer as UpstreamVoxelizer


class FusedModelEma(UpstreamModelEma):
    """Use the repository's fused multi-tensor EMA update."""

    def __init__(self, model, decay=0.9999, device=None, foreach=True):
        super().__init__(
            model,
            decay=decay,
            device=device,
            foreach=True,
        )


class ReducedSyncVoxelizer(UpstreamVoxelizer):
    """Run batch-32 PyUUL voxelization in two chunks instead of four."""

    def mol2vox(self, batch: list, num_channels: int = 7) -> torch.Tensor:
        batch = {
            "coords": batch["coords"].to(self.device, non_blocking=True),
            "radius": batch["radius"].to(self.device, non_blocking=True),
            "atoms_channel": batch["atoms_channel"].to(
                self.device, non_blocking=True
            ),
        }

        if self.backend == "torch":
            return self._torch_voxelize(batch, num_channels)

        batch = self._add_dumb_coords(batch)
        batch_size = batch["coords"].shape[0]
        requested_chunks = int(
            os.environ.get("VOXBIND_PYUUL_CHUNKS", "2")
        )
        n_chunks = requested_chunks if batch_size > 16 else 1
        n_chunks = max(1, min(n_chunks, batch_size))
        chunk_size = (batch_size + n_chunks - 1) // n_chunks
        voxels = torch.empty(
            (
                batch_size,
                num_channels,
                self.grid_dim,
                self.grid_dim,
                self.grid_dim,
            ),
            device=self.device,
        )

        for chunk_index in range(n_chunks):
            start = chunk_index * chunk_size
            end = min((chunk_index + 1) * chunk_size, batch_size)
            if start >= end:
                break
            chunk_voxels = self.vol_maker(
                batch["coords"][start:end],
                batch["radius"][start:end],
                batch["atoms_channel"][start:end],
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=num_channels,
            )
            center = chunk_voxels.shape[-1] // 2
            box_min = center - self.grid_dim // 2
            box_max = center + self.grid_dim // 2
            voxels[start:end] = chunk_voxels[
                :,
                :,
                box_min:box_max,
                box_min:box_max,
                box_min:box_max,
            ]
            del chunk_voxels

        return voxels


upstream.ModelEma = FusedModelEma
upstream.Voxelizer = ReducedSyncVoxelizer


if __name__ == "__main__":
    print(
        "[optimized-entry] fused EMA enabled; "
        f"PyUUL chunks={os.environ.get('VOXBIND_PYUUL_CHUNKS', '2')}; "
        "AdamW unchanged",
        flush=True,
    )
    upstream.main()
