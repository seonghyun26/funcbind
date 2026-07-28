"""Channel-wise Gaussian splat decoder for FuncBind neural fields.

The decoder keeps the released encoder and latent grid unchanged.  Every latent
voxel predicts ``K`` isotropic Gaussians for every atom channel.  Rendering uses
the same probabilistic-union occupancy convention as :class:`FieldMaker`:

    occupancy = 1 - product(1 - opacity * gaussian_density)

Two renderers are provided:

* :func:`render_gaussians_chunked` is a general differentiable all-pairs
  reference renderer.
* :class:`ChannelWiseGaussianSplatDecoder3D` uses a bounded local neighborhood
  around each query point, which makes training on 128^3 fields affordable.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from funcbind.models.decoder import (
    Decoder,
    _normalize_coords,
    _unnormalize_coords,
    get_atom_coords_batched,
    get_grid,
    unrotate_gen_mols,
)


def _validate_renderer_inputs(
    query_points: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
) -> None:
    if query_points.ndim != 3 or query_points.shape[-1] != 3:
        raise ValueError("query_points must have shape [batch, points, 3]")
    if centers.ndim != 4 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape [batch, channels, gaussians, 3]")
    if scales.ndim == 4 and scales.shape[-1] == 1:
        scales = scales.squeeze(-1)
    if scales.shape != centers.shape[:-1]:
        raise ValueError("scales must have shape [batch, channels, gaussians]")
    if opacities.shape != centers.shape[:-1]:
        raise ValueError("opacities must have shape [batch, channels, gaussians]")
    if query_points.shape[0] != centers.shape[0]:
        raise ValueError("query_points and centers must have the same batch size")


def render_gaussians_chunked(
    query_points: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    *,
    query_chunk_size: int = 1024,
    gaussian_chunk_size: int = 512,
    opacity_threshold: float = 0.0,
    cutoff_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Render channel-wise isotropic Gaussians at arbitrary 3D query points.

    Args:
        query_points: Normalized coordinates with shape ``[B, Q, 3]``.
        centers: Gaussian centers with shape ``[B, C, G, 3]``.
        scales: Positive isotropic scales with shape ``[B, C, G]``.
        opacities: Values in ``[0, 1]`` with shape ``[B, C, G]``.
        query_chunk_size: Maximum query points evaluated together.
        gaussian_chunk_size: Maximum Gaussians evaluated together.
        opacity_threshold: Gaussians below this opacity are ignored.
        cutoff_sigma: Optional compact-support cutoff measured in standard
            deviations.  The Gaussian remains differentiable inside the cutoff.

    Returns:
        Occupancies with shape ``[B, Q, C]``.
    """
    if scales.ndim == 4 and scales.shape[-1] == 1:
        scales = scales.squeeze(-1)
    _validate_renderer_inputs(query_points, centers, scales, opacities)
    if query_chunk_size <= 0 or gaussian_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    output_dtype = query_points.dtype
    work_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    query_points = query_points.to(work_dtype)
    centers = centers.to(work_dtype)
    scales = scales.to(work_dtype).clamp_min(eps)
    opacities = opacities.to(work_dtype).clamp(0.0, 1.0 - eps)

    rendered = []
    for query_chunk in query_points.split(query_chunk_size, dim=1):
        batch, n_query, _ = query_chunk.shape
        n_channels = centers.shape[1]
        log_survival = torch.zeros(
            batch,
            n_query,
            n_channels,
            device=query_chunk.device,
            dtype=work_dtype,
        )
        for start in range(0, centers.shape[2], gaussian_chunk_size):
            stop = min(start + gaussian_chunk_size, centers.shape[2])
            center_chunk = centers[:, :, start:stop]
            scale_chunk = scales[:, :, start:stop]
            opacity_chunk = opacities[:, :, start:stop]

            delta = (
                query_chunk[:, :, None, None, :]
                - center_chunk[:, None, :, :, :]
            )
            distance_sq = delta.square().sum(dim=-1)
            scale_sq = scale_chunk[:, None, :, :].square().clamp_min(eps)
            density = opacity_chunk[:, None, :, :] * torch.exp(
                -distance_sq / scale_sq
            )
            if cutoff_sigma is not None:
                within_cutoff = distance_sq <= (
                    cutoff_sigma * scale_chunk[:, None, :, :]
                ).square()
                density = density * within_cutoff
            if opacity_threshold > 0:
                density = density * (
                    opacity_chunk[:, None, :, :] >= opacity_threshold
                )
            density = density.clamp(0.0, 1.0 - eps)
            log_survival = log_survival + torch.log1p(-density).sum(dim=-1)

        rendered.append((1.0 - torch.exp(log_survival)).to(output_dtype))
    return torch.cat(rendered, dim=1)


class ChannelWiseGaussianSplatDecoder3D(Decoder):
    """Predict and render channel-wise Gaussian density bases.

    ``K`` Gaussians are emitted by each latent voxel for each atom-type channel.
    Centers are bounded to a configurable fraction of a latent cell, scales are
    bounded in Angstrom, and opacities use a sigmoid parameterization.
    """

    def __init__(
        self,
        n_channels: int,
        code_dim: int,
        grid_dim: int,
        latent_grid_dim: int,
        *,
        hidden_dim: int = 512,
        gaussians_per_voxel: int = 1,
        scale_min: float = 0.35,
        scale_max: float = 1.25,
        offset_bound: float = 0.5,
        opacity_threshold: float = 0.01,
        initial_opacity: float = 0.05,
        cutoff_sigma: float = 3.0,
        query_chunk_size: int = 512,
        gaussian_chunk_size: int = 512,
        resolution: float = 0.25,
        fabric=None,
    ) -> None:
        # Decoder is inherited for its molecule/normalization API, but its INR
        # network is intentionally not constructed.
        nn.Module.__init__(self)
        if n_channels <= 0 or code_dim <= 0:
            raise ValueError("n_channels and code_dim must be positive")
        if gaussians_per_voxel <= 0:
            raise ValueError("gaussians_per_voxel must be positive")
        if not 0 < scale_min <= scale_max:
            raise ValueError("expected 0 < scale_min <= scale_max")
        if not 0 <= offset_bound <= 1:
            raise ValueError("offset_bound must be in [0, 1] latent-cell units")
        if not 0 <= opacity_threshold < 1:
            raise ValueError("opacity_threshold must be in [0, 1)")
        if not 0 < initial_opacity < 1:
            raise ValueError("initial_opacity must be in (0, 1)")

        self.n_channels = n_channels
        self.code_dim = code_dim
        self.grid_dim = grid_dim
        self.latent_grid_dim = latent_grid_dim
        self.hidden_dim = hidden_dim
        self.gaussians_per_voxel = gaussians_per_voxel
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.offset_bound = float(offset_bound)
        self.opacity_threshold = float(opacity_threshold)
        self.cutoff_sigma = float(cutoff_sigma)
        self.query_chunk_size = int(query_chunk_size)
        self.gaussian_chunk_size = int(gaussian_chunk_size)
        self.resolution = float(resolution)
        self.fabric = fabric
        self.reshape_out = 1
        self.per_patch_coord = False
        self.coord_dim = 3

        output_channels = n_channels * gaussians_per_voxel * 5
        self.parameter_head = nn.Sequential(
            nn.Conv3d(code_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, output_channels, kernel_size=1),
        )
        final = self.parameter_head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            bias = final.bias.view(n_channels, gaussians_per_voxel, 5)
            opacity_logit = math.log(initial_opacity / (1.0 - initial_opacity))
            bias[..., 4].fill_(opacity_logit)

        device = getattr(fabric, "device", torch.device("cpu"))
        self.coords = get_grid(grid_dim)[1].unsqueeze(0).to(device)
        if fabric is not None:
            fabric.print(">> coords shape:", self.coords.shape)

        cell_spacing = 2.0 / max(latent_grid_dim - 1, 1)
        domain_half_width = resolution * grid_dim / 2.0
        max_scale_normalized = self.scale_max / domain_half_width
        local_radius = math.ceil(
            (
                self.cutoff_sigma * max_scale_normalized
                + self.offset_bound * cell_spacing
            )
            / cell_spacing
        )
        local_radius = max(local_radius, 1)
        offsets = torch.cartesian_prod(
            *[
                torch.arange(-local_radius, local_radius + 1, dtype=torch.long)
                for _ in range(3)
            ]
        )
        self.register_buffer("neighbor_offsets", offsets, persistent=False)

    @property
    def decoder_type(self) -> str:
        return "gaussian_splat"

    @property
    def requires_full_latent_grid(self) -> bool:
        return True

    def _latent_anchors(
        self,
        spatial_shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        axes = [
            torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
            for size in spatial_shape
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
            -1, 3
        )

    def decode_parameters(self, codes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert a latent grid into bounded Gaussian parameters."""
        if codes.ndim != 5:
            raise ValueError(
                "Gaussian decoder requires a full latent grid [B, D, Sx, Sy, Sz]; "
                f"received {tuple(codes.shape)}"
            )
        raw = self.parameter_head(codes)
        batch, _, sx, sy, sz = raw.shape
        raw = raw.view(
            batch,
            self.n_channels,
            self.gaussians_per_voxel,
            5,
            sx,
            sy,
            sz,
        ).permute(0, 4, 5, 6, 1, 2, 3)
        raw = raw.reshape(
            batch,
            sx * sy * sz,
            self.n_channels,
            self.gaussians_per_voxel,
            5,
        )

        anchors = self._latent_anchors(
            (sx, sy, sz), device=raw.device, dtype=raw.dtype
        )
        spacing = raw.new_tensor(
            [
                2.0 / max(sx - 1, 1),
                2.0 / max(sy - 1, 1),
                2.0 / max(sz - 1, 1),
            ]
        )
        offsets = torch.tanh(raw[..., :3]) * (
            self.offset_bound * spacing
        )
        centers = anchors[None, :, None, None, :] + offsets

        scale_angstrom = self.scale_min + (
            self.scale_max - self.scale_min
        ) * torch.sigmoid(raw[..., 3])
        domain_half_width = self.resolution * self.grid_dim / 2.0
        scales = scale_angstrom / domain_half_width
        opacities = torch.sigmoid(raw[..., 4])
        return {
            "centers": centers,
            "scales": scales,
            "opacities": opacities,
        }

    def _render_local(
        self,
        query_points: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        *,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor:
        centers = parameters["centers"]
        scales = parameters["scales"]
        opacities = parameters["opacities"]
        batch, n_voxels, _, _, _ = centers.shape
        if query_points.shape[0] != batch:
            raise ValueError("query and latent-grid batch sizes differ")

        spatial_size = round(n_voxels ** (1.0 / 3.0))
        if spatial_size ** 3 != n_voxels:
            raise ValueError("local renderer currently requires a cubic latent grid")
        chunk_size = query_chunk_size or self.query_chunk_size
        output_dtype = query_points.dtype
        work_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        centers = centers.to(work_dtype)
        scales = scales.to(work_dtype).clamp_min(1e-6)
        opacities = opacities.to(work_dtype)
        offsets = self.neighbor_offsets.to(query_points.device)
        batch_indices = torch.arange(batch, device=query_points.device)[
            :, None, None
        ]

        outputs = []
        for query_chunk in query_points.split(chunk_size, dim=1):
            query_chunk = query_chunk.to(work_dtype)
            base = torch.round(
                (query_chunk + 1.0) * (spatial_size - 1) / 2.0
            ).long()
            neighbor_indices = base[:, :, None, :] + offsets[None, None, :, :]
            valid = (
                (neighbor_indices >= 0)
                & (neighbor_indices < spatial_size)
            ).all(dim=-1)
            neighbor_indices = neighbor_indices.clamp(0, spatial_size - 1)
            linear = (
                neighbor_indices[..., 0] * spatial_size * spatial_size
                + neighbor_indices[..., 1] * spatial_size
                + neighbor_indices[..., 2]
            )

            local_centers = centers[batch_indices, linear]
            local_scales = scales[batch_indices, linear]
            local_opacities = opacities[batch_indices, linear]
            delta = (
                query_chunk[:, :, None, None, None, :]
                - local_centers
            )
            distance_sq = delta.square().sum(dim=-1)
            density = local_opacities * torch.exp(
                -distance_sq / local_scales.square().clamp_min(1e-6)
            )
            density = density * valid[:, :, :, None, None]
            density = density * (
                distance_sq
                <= (self.cutoff_sigma * local_scales).square()
            )
            if not self.training and self.opacity_threshold > 0:
                density = density * (
                    local_opacities >= self.opacity_threshold
                )
            density = density.clamp(0.0, 1.0 - 1e-6)
            # [B, Q, local_voxels, channels, K] -> [B, Q, channels]
            log_survival = torch.log1p(-density).sum(dim=(2, 4))
            outputs.append(
                (1.0 - torch.exp(log_survival)).to(output_dtype)
            )
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        parameters = self.decode_parameters(codes)
        return self._render_local(x, parameters)

    def forward_batched(
        self,
        xs: torch.Tensor,
        codes: torch.Tensor,
        batch_size_render: int = 100_000,
        threshold: float | None = None,
        to_cpu: bool = True,
    ) -> torch.Tensor:
        parameters = self.decode_parameters(codes)
        chunk_size = min(batch_size_render, self.query_chunk_size)
        predictions = []
        for x_chunk in xs.split(chunk_size, dim=1):
            prediction = self._render_local(
                x_chunk, parameters, query_chunk_size=chunk_size
            )
            if threshold is not None:
                prediction = torch.where(
                    prediction >= threshold, prediction, 0
                )
            predictions.append(prediction.cpu() if to_cpu else prediction)
        return torch.cat(predictions, dim=1)

    def render_code(
        self,
        codes: torch.Tensor,
        batch_size_render: int,
        fabric=None,
        verbose: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            if verbose and fabric is not None:
                fabric.print(
                    f">> Rendering Gaussian splats - query chunks of "
                    f"{min(batch_size_render, self.query_chunk_size)}"
                )
            device = next(self.parameters()).device
            codes = codes.to(device)
            xs = self.coords.to(device)
            if xs.shape[0] != codes.shape[0]:
                xs = xs.expand(codes.shape[0], -1, -1)
            prediction = self.forward_batched(
                xs,
                codes,
                batch_size_render=batch_size_render,
                to_cpu=True,
            )
            return prediction.permute(0, 2, 1).reshape(
                -1,
                self.n_channels,
                self.grid_dim,
                self.grid_dim,
                self.grid_dim,
            )

    def codes_to_molecules(
        self,
        codes,
        unnormalize,
        config,
        fabric=None,
        rand_rots=None,
        verbose=True,
        target_dirname=None,
        voxel_name=None,
    ):
        """Render, peak-extract, and return molecules without INR refinement."""
        codes = codes.detach()
        if unnormalize:
            codes = self.unnormalize_code(codes).detach()
        grids = self.render_code(
            codes,
            config["sampling"]["batch_size_render"],
            fabric=fabric,
            verbose=verbose,
        )
        mols = get_atom_coords_batched(
            grids,
            fabric,
            rad=config["dset"].get("ligand_radius", 0.5),
            resolution=self.resolution,
            flexible=config["sampling"].get("flexible", True),
        )
        decoded = []
        rotations = []
        for index, molecule in enumerate(mols):
            if molecule is None:
                continue
            molecule = _normalize_coords(molecule, self.grid_dim)
            molecule = _unnormalize_coords(
                molecule, self.grid_dim, self.resolution
            )
            decoded.append(molecule)
            if rand_rots is not None:
                rotations.append(
                    rand_rots[index] if len(rand_rots) > 1 else rand_rots[0]
                )
        if rotations:
            decoded = unrotate_gen_mols(decoded, rotations, fabric)
        return decoded
