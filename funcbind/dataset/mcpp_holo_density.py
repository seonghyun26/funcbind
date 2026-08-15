"""Fast receptor-level holo 2Fo-Fc density access for the MCP dataset.

The preprocessing job stores one raw, canonical-frame density box per deposited
receptor.  Every MCP conformer for that receptor reuses the same box.  At read time
we crop around the current conformer's centre and apply exactly the same rotation and
translation as the atoms before applying the VoxBind pretraining normalization.

The source maps are deliberately *holo*: density from the deposited reference ligand
is retained.  This module never erases, masks, or synthesizes density.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import torch


GRID_DIM = 64
RESOLUTION = 0.25


def mcpp_target_id(mcp_path: str) -> str:
    """Return the four-character receptor target directory for an MCP sample."""
    parts = Path(mcp_path).parts
    for part in reversed(parts[:-1]):
        candidate = part.lower()
        if len(candidate) == 4 and candidate.isalnum():
            return candidate
    raise ValueError(f"cannot infer MCP target id from {mcp_path!r}")


def _apply_normalization(raw: np.ndarray, norm: dict) -> np.ndarray:
    scheme = str(norm.get("scheme", "")).lower()
    if "arcsinh" not in scheme:
        raise ValueError(
            "MCP holo-density requires the canonical VoxBind arcsinh normalization; "
            f"got scheme={norm.get('scheme')!r}"
        )
    scale = float(norm["arcsinh_scale"])
    mu = float(norm["mu_a"])
    sigma = float(norm["sigma_a"])
    if scale <= 0 or sigma <= 0:
        raise ValueError(f"invalid density normalization: scale={scale}, sigma={sigma}")
    return (np.arcsinh(raw.astype(np.float32) / scale) - mu) / sigma


def resample_holo_box(
    box: np.ndarray,
    center_delta: np.ndarray,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    *,
    grid_dim: int = GRID_DIM,
    resolution: float = RESOLUTION,
) -> Optional[np.ndarray]:
    """Crop density aligned to a recentered and optionally augmented MCP conformer.

    ``box`` is in the original MCP receptor frame and centred on the deposited
    reference ligand. ``center_delta`` is current conformer centre minus reference
    centre in that same frame. FuncBind augments row-vector coordinates as
    ``p -> p @ rotation.T + translation``; inverse augmentation is therefore applied
    to each output offset before indexing the canonical box.

    Returns ``None`` instead of a partially zero-padded crop whenever the requested
    field of view falls outside the cached box. The caller then marks the sample as
    unavailable, guaranteeing an exact density-free residual for that example.
    """
    g_box = int(box.shape[0])
    if box.shape != (g_box, g_box, g_box):
        raise ValueError(f"density box must be cubic, got {box.shape}")

    off = (np.arange(grid_dim, dtype=np.float32) - grid_dim * 0.5) * resolution
    gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
    offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    if rotation is not None:
        if translation is not None:
            offsets -= np.asarray(translation, dtype=np.float32).reshape(1, 3)
        offsets = offsets @ np.asarray(rotation, dtype=np.float32)

    relative = offsets + np.asarray(center_delta, dtype=np.float32).reshape(1, 3)
    index = relative / resolution + g_box * 0.5
    # A partially padded crop creates an augmentation-dependent edge artefact. It is
    # safer to turn density off for this rare, far-displaced conformer.
    if float(index.min()) < 0.0 or float(index.max()) > g_box - 1:
        return None

    density = scipy.ndimage.map_coordinates(
        box,
        [index[:, 0], index[:, 1], index[:, 2]],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return density.reshape(grid_dim, grid_dim, grid_dim).astype(np.float32)


class MCPPHoloDensityStore:
    """Lazy, fork-safe reader for target-level MCP holo-density boxes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        manifest_path = self.root / "manifest.json"
        if not meta_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"incomplete MCP holo-density dataset at {self.root}; "
                "expected meta.json and manifest.json"
            )

        self.meta = json.loads(meta_path.read_text())
        records = json.loads(manifest_path.read_text())
        self.records = {record["target_id"].lower(): record for record in records}
        self.g_box = int(self.meta["g_box"])
        self.resolution = float(self.meta["resolution"])
        self.n_boxes = int(self.meta["n_boxes"])
        self.dtype = np.dtype(self.meta.get("dtype", "float16"))
        self.normalization = self.meta["normalization"]
        self.box_path = self.root / self.meta.get("box_file", "boxes_float16.dat")
        expected_bytes = self.n_boxes * self.g_box ** 3 * self.dtype.itemsize
        if not self.box_path.exists() or self.box_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"density box file is missing or incomplete: {self.box_path} "
                f"(expected {expected_bytes:,} bytes)"
            )
        self._boxes_memmap = None

    def _boxes(self):
        if self._boxes_memmap is None:
            self._boxes_memmap = np.memmap(
                self.box_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.n_boxes, self.g_box, self.g_box, self.g_box),
            )
        return self._boxes_memmap

    def available(self, target_id: str) -> bool:
        record = self.records.get(target_id.lower())
        return bool(record is not None and int(record.get("box_index", -1)) >= 0)

    def reference_ligand(self, target_id: str) -> str:
        record = self.records.get(target_id.lower(), {})
        return str(record.get("reference_ligand", ""))

    def load(
        self,
        target_id: str,
        sample_center: torch.Tensor | np.ndarray,
        rotation: Optional[torch.Tensor | np.ndarray] = None,
        translation: Optional[torch.Tensor | np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records.get(target_id.lower())
        blank = torch.zeros(GRID_DIM, GRID_DIM, GRID_DIM, dtype=torch.float32)
        if record is None or int(record.get("box_index", -1)) < 0:
            return blank, torch.tensor(False)

        def as_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        center = as_numpy(sample_center).reshape(3)
        reference_center = np.asarray(record["reference_center"], dtype=np.float32)
        center_delta = center - reference_center
        raw_box = np.asarray(self._boxes()[int(record["box_index"])], dtype=np.float32)
        crop = resample_holo_box(
            raw_box,
            center_delta,
            rotation=as_numpy(rotation),
            translation=as_numpy(translation),
            grid_dim=GRID_DIM,
            resolution=self.resolution,
        )
        if crop is None or not np.isfinite(crop).all():
            return blank, torch.tensor(False)
        crop = _apply_normalization(crop, self.normalization)
        return torch.from_numpy(crop), torch.tensor(True)
