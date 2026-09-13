"""Frozen VoxBind CDG encoder as a zero-init conditioning branch for FuncBind.

The frozen encoder feeds the RECEPTOR representation through a zero-initialised
projection, so step 0 is bit-identical to the density-free model and the gradient is
still non-zero (the input is not zero, only the projection is).

FUSION MODES. What separates VoxBind's fusion variants is not where the residual is
added -- adding it to the pocket and then summing with the ligand is the same arithmetic
as adding it to the sum -- but WHAT THE PROJECTION IS CONDITIONED ON:

    density_only    proj(dens)                      the original FuncBind branch
    protein_first   proj([receptor, dens])          VoxBind's protein_first
    default         proj([receptor + ligand, dens]) VoxBind's default

`default` is the variant that beat the density-free baseline on the 78-pocket CrossDocked
benchmark (-8.45 vs -8.05); it is the only one whose correction can depend on the noisy
generation target. `density_only` was this file's original behaviour, described in its
first version as mirroring `protein_first` -- it does not: it sees neither the receptor
nor the ligand. It is kept so the 3.17M / 8.21M / 26.1M checkpoints stay reproducible.

Under `default` the ligand term is the PRECONDITIONED input c_in * ligand_encoding, which
is what the UNet itself consumes, so the branch sees the ligand at the same scale the
denoiser does. That forces the fusion to happen after preconditioning -- see denoiser.py.

The encoder is the same 13-channel ChannelViT used by VoxBind:
    [7 ligand channels (all zero — unknown at generation time),
     4 receptor atom-blob channels,
     experimental 2Fo-Fc density,
     gradient magnitude]

Everything here is inert unless `denoiser.with_density=true`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# vdW radii for the 4 receptor atom-blob channels (C, O, N, S), matching the radii the
# density encoder was pretrained with. FuncBind stores its own per-atom radii, which are
# NOT the same convention, so they are overwritten before voxelisation.
RECEPTOR_VDW = (1.70, 1.52, 1.55, 1.80)

_N_LIGAND_CH = 7
_N_RECEPTOR_CH = 4
_GRID = 64
_RES = 0.25


def _add_voxbind_to_path(voxbind_root: str | Path) -> None:
    root = str(Path(voxbind_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_density_encoder(cfg, voxbind_root: str | Path, device=None):
    """Load a frozen VoxBind DensityViT from a pretraining checkpoint.

    `cfg` carries the ViT geometry, which differs per checkpoint (atomblob7 v2.1 is
    dim 512 / depth 12 / groups [7,4,1,1]; the champion is dim 640 / depth 18 /
    groups [7,4,2]), so nothing here is hard-coded.
    """
    _add_voxbind_to_path(voxbind_root)
    from voxbind.models.density_vit import DensityViT

    encoder = DensityViT(
        grid_dim=_GRID,
        patch_size=int(cfg.get("patch", 8)),
        n_in_channels=13,
        c_out=int(cfg.get("c_out", 16)),
        dim=int(cfg["dim"]),
        depth=int(cfg["depth"]),
        n_heads=int(cfg["heads"]),
        mlp_ratio=int(cfg.get("mlp_ratio", 4)),
        dropout=0.0,
        pos_encoding=cfg.get("pos_encoding", "learnable"),
        patch_embed_mode=cfg.get("patch_embed_mode", "channel_group"),
        channel_groups=tuple(cfg["channel_groups"]),
        channel_group_dropout=0.0,
        n_memory_tokens=int(cfg.get("n_memory_tokens", 0)),
    )

    ckpt = torch.load(str(cfg["pretrained_path"]), map_location="cpu",
                      weights_only=False, mmap=True)
    raw = ckpt.get("encoder_state_dict_ema") or ckpt["encoder_state_dict"]
    stripped = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(stripped or raw, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"density encoder state_dict mismatch: missing={sorted(missing)[:6]} "
            f"unexpected={sorted(unexpected)[:6]}"
        )
    del ckpt

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    if device is not None:
        encoder.to(device)
    return encoder


class DensityCondition(nn.Module):
    """Frozen CDG encoder → zero-init projection → residual on the receptor condition.

    forward() returns a delta shaped like `receptor_encoding` (B, code_dim, G, G, G);
    at initialisation that delta is exactly zero.
    """

    def __init__(
        self,
        density_cfg,
        code_dim: int,
        code_grid_dim: int,
        voxbind_root: str | Path,
        hidden: int = 192,
        freeze: bool = True,
        amp: bool = True,
        latent_extent: float | None = None,
        fusion: str = "density_only",
    ):
        super().__init__()
        self.encoder = build_density_encoder(density_cfg, voxbind_root)
        self.freeze = freeze
        self.amp = amp
        self.code_grid_dim = code_grid_dim
        # Physical size of FuncBind's receptor latent box, in Angstrom (grid_dim *
        # resolution = 128 * 0.25 = 32 A). The density crop is a different physical
        # size (64 * 0.25 = 16 A), so the two grids are only comparable through their
        # cell pitch — see forward(). None keeps the pre-registration behaviour.
        self.latent_extent = float(latent_extent) if latent_extent else None
        self.density_extent = _GRID * _RES
        dim = int(density_cfg["dim"])

        # Validate explicitly rather than falling through to a default: a typo here would
        # silently train a different experiment for weeks, which is exactly the failure
        # this run already had once.
        _valid = ("density_only", "protein_first", "default")
        self.fusion = str(fusion)
        if self.fusion not in _valid:
            raise ValueError(f"density fusion must be one of {_valid}, got {fusion!r}")
        self.code_dim = int(code_dim)
        # The conditioned modes concatenate a code_dim-wide tensor onto the encoder
        # features before the projection; density_only keeps the original width.
        in_ch = dim + (0 if self.fusion == "density_only" else self.code_dim)

        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, code_dim, kernel_size=1),
        )
        # ControlNet-style zero convolution: the branch starts as a no-op.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()  # keep the frozen trunk in eval regardless of the parent
        return self

    @torch.no_grad()
    def _encode(self, density_input: torch.Tensor) -> torch.Tensor:
        """13-channel volume → (B, dim, g, g, g) feature map, under no_grad (frozen)."""
        enc = self.encoder
        # amp=False means "run this trunk in fp32", but the caller sits inside Fabric's
        # bf16 autocast and hands us a bf16 volume. Disabling autocast without casting the
        # input leaves bf16 activations meeting fp32 weights, which is a hard error
        # ("Input type (c10::BFloat16) and bias type (float) should be the same") on the
        # very first batch. Match the input to the frozen weights instead.
        if not self.amp:
            enc_dtype = next(enc.parameters()).dtype
            if density_input.dtype != enc_dtype:
                density_input = density_input.to(enc_dtype)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.amp and density_input.is_cuda
        ):
            tokens = enc.forward_features(density_input)
            pooled = enc._pool_groups(tokens)          # (B, N, dim), N = g**3
        pooled = pooled.float()
        b, n, d = pooled.shape
        g = round(n ** (1 / 3))
        if g ** 3 != n:
            raise RuntimeError(f"token count {n} is not a cube; cannot reshape to a grid")
        return pooled.transpose(1, 2).reshape(b, d, g, g, g)

    def _n_latent_cells(self) -> int:
        """How many receptor-latent cells the density crop physically covers.

        Both grids are centred on the same point (the recentred conformer), so the
        density can only be added where it was actually measured. With the shipped
        geometry the density patch pitch (8 voxels * 0.25 A = 2 A) equals the latent
        cell pitch (32 A / 16 = 2 A), and 16 A of density covers 8 of the 16 cells.
        Resizing the 8-cell feature map up to the full 16-cell grid instead — as an
        unregistered interpolate() does — would double every feature's distance from
        the centre and put the density in the wrong place.
        """
        cell = self.latent_extent / self.code_grid_dim
        n = int(round(self.density_extent / cell))
        return max(1, min(n, self.code_grid_dim))

    def _crop_cond(self, cond: torch.Tensor, n_cells: int) -> torch.Tensor:
        """The central n_cells^3 of a code_grid_dim^3 conditioning tensor.

        The projection runs on the cells the density crop actually covers, so whatever is
        concatenated onto it has to be cropped to the same window -- otherwise the two
        halves of the concatenation would describe different regions of the pocket.
        """
        if cond.shape[-1] == n_cells:
            return cond
        lo = (self.code_grid_dim - n_cells) // 2
        return cond[:, :, lo:lo + n_cells, lo:lo + n_cells, lo:lo + n_cells]

    def _with_cond(self, feat: torch.Tensor, cond: torch.Tensor | None,
                   n_cells: int) -> torch.Tensor:
        if self.fusion == "density_only":
            return feat
        if cond is None:
            raise ValueError(
                f"density fusion {self.fusion!r} needs a conditioning tensor; "
                "the caller passed none"
            )
        if cond.shape[1] != self.code_dim:
            raise ValueError(
                f"conditioning tensor has {cond.shape[1]} channels, expected {self.code_dim}"
            )
        return torch.cat([self._crop_cond(cond, n_cells).to(feat.dtype), feat], dim=1)

    def forward(self, density_input: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        feat = self._encode(density_input)
        if self.latent_extent is None:
            if feat.shape[-1] != self.code_grid_dim:
                feat = F.interpolate(
                    feat, size=(self.code_grid_dim,) * 3, mode="trilinear",
                    align_corners=False,
                )
            return self.proj(self._with_cond(feat, cond, self.code_grid_dim))

        n_cells = self._n_latent_cells()
        if feat.shape[-1] != n_cells:
            feat = F.interpolate(
                feat, size=(n_cells,) * 3, mode="trilinear", align_corners=False
            )
        delta = self.proj(self._with_cond(feat, cond, n_cells))
        if n_cells == self.code_grid_dim:
            return delta
        # Project first, then place: the padding stays exactly zero instead of picking
        # up the projection's bias, so the branch cannot invent a density signal in the
        # region the crop never saw.
        out = delta.new_zeros(
            delta.shape[0], delta.shape[1], *(self.code_grid_dim,) * 3
        )
        lo = (self.code_grid_dim - n_cells) // 2
        out[:, :, lo:lo + n_cells, lo:lo + n_cells, lo:lo + n_cells] = delta
        return out


def apply_density_residual(
    receptor_encoding: torch.Tensor,
    density_delta: torch.Tensor,
    density_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a residual with an exact per-sample no-op for unavailable maps."""
    if density_available is not None:
        available = torch.as_tensor(
            density_available,
            device=density_delta.device,
            dtype=density_delta.dtype,
        )
        if available.ndim == 0:
            available = available.reshape(1)
        if available.numel() not in (1, density_delta.shape[0]):
            raise ValueError(
                f"density availability has {available.numel()} rows for "
                f"a density batch of {density_delta.shape[0]}"
            )
        density_delta = density_delta * available.reshape(-1, 1, 1, 1, 1)
    if density_delta.shape[0] not in (1, receptor_encoding.shape[0]):
        raise ValueError(
            f"density batch {density_delta.shape[0]} cannot condition receptor batch "
            f"{receptor_encoding.shape[0]}"
        )
    return receptor_encoding + density_delta


def build_density_input(
    receptor: dict,
    density: torch.Tensor,
    voxelizer,
    voxbind_root: str | Path,
) -> torch.Tensor:
    """Assemble the 13-channel encoder input for a BATCH.

    `receptor` is FuncBind's receptor dict (coords / atoms_channel / radius); `density`
    is (B, 1, 64, 64, 64) already normalised by the v5 crop statistics. The 7 ligand
    channels stay zero because the ligand is unknown at generation time — the same
    condition VoxBind trains and samples under.
    """
    _add_voxbind_to_path(voxbind_root)
    from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore

    if density.dim() == 4:
        density = density.unsqueeze(1)
    gradmag = per_sample_zscore(gradient_magnitude3d(density))

    rec = {
        "coords": receptor["coords"],
        "atoms_channel": receptor["atoms_channel"],
        "radius": receptor["radius"].clone(),
    }
    channel = rec["atoms_channel"].long()
    lut = torch.tensor(RECEPTOR_VDW, device=channel.device, dtype=rec["radius"].dtype)
    valid = (channel >= 0) & (channel < _N_RECEPTOR_CH)
    rec["radius"][valid] = lut[channel[valid]]
    rec["radius"][~valid] = 0.5

    pocket = voxelizer(rec, num_channels=_N_RECEPTOR_CH)
    b = pocket.shape[0]
    ligand_unknown = torch.zeros(
        (b, _N_LIGAND_CH, _GRID, _GRID, _GRID), device=pocket.device, dtype=pocket.dtype
    )
    out = torch.cat(
        [ligand_unknown, pocket, density.to(pocket.dtype), gradmag.to(pocket.dtype)], dim=1
    )
    if out.shape[1:] != (13, _GRID, _GRID, _GRID):
        raise RuntimeError(f"unexpected density input shape {tuple(out.shape)}")
    return out


def make_density_voxelizer(device, voxbind_root: str | Path, backend: str = "torch"):
    """The 64³ / 0.25 Å voxelizer the density encoder was pretrained against.

    FuncBind's own receptor grid is 128³, so this is deliberately a separate instance.
    """
    _add_voxbind_to_path(voxbind_root)
    from voxbind.voxelizer import Voxelizer

    return Voxelizer(
        grid_dim=_GRID, resolution=_RES, radius=0.5, cubes_around=8,
        device=str(device), backend=backend,
    )
