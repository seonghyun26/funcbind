from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from funcbind.models.gaussian_splat_decoder import (
    ChannelWiseGaussianSplatDecoder3D,
    render_gaussians_chunked,
)
from funcbind.utils.utils_nf import create_nf_decoder, infer_codes_batch


class FabricStub:
    device = torch.device("cpu")

    @staticmethod
    def print(*args, **kwargs):
        return None


def _flatten_parameters(parameters):
    centers = parameters["centers"]
    scales = parameters["scales"]
    opacities = parameters["opacities"]
    batch, _, channels, _, _ = centers.shape
    return (
        centers.permute(0, 2, 1, 3, 4).reshape(batch, channels, -1, 3),
        scales.permute(0, 2, 1, 3).reshape(batch, channels, -1),
        opacities.permute(0, 2, 1, 3).reshape(batch, channels, -1),
    )


def test_reference_renderer_matches_single_gaussian_analytic_values():
    query_points = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    centers = torch.zeros(1, 1, 1, 3)
    scales = torch.full((1, 1, 1), 0.2)
    opacities = torch.full((1, 1, 1), 0.4)

    rendered = render_gaussians_chunked(
        query_points,
        centers,
        scales,
        opacities,
        query_chunk_size=1,
        gaussian_chunk_size=1,
    )

    expected = torch.tensor([[[0.4], [0.4 * torch.exp(torch.tensor(-1.0))]]])
    torch.testing.assert_close(rendered, expected)


def test_reference_renderer_is_chunk_invariant_and_differentiable():
    torch.manual_seed(4)
    query_points = (torch.rand(2, 11, 3) * 2 - 1).requires_grad_()
    centers = (torch.rand(2, 3, 7, 3) * 2 - 1).requires_grad_()
    scales = torch.rand(2, 3, 7, requires_grad=True) * 0.2 + 0.1
    opacities = torch.sigmoid(torch.randn(2, 3, 7, requires_grad=True))

    unchunked = render_gaussians_chunked(
        query_points,
        centers,
        scales,
        opacities,
        query_chunk_size=64,
        gaussian_chunk_size=64,
    )
    chunked = render_gaussians_chunked(
        query_points,
        centers,
        scales,
        opacities,
        query_chunk_size=3,
        gaussian_chunk_size=2,
    )

    torch.testing.assert_close(chunked, unchunked, atol=2e-6, rtol=2e-6)
    chunked.sum().backward()
    assert query_points.grad is not None and torch.isfinite(query_points.grad).all()
    assert centers.grad is not None and torch.isfinite(centers.grad).all()


def test_local_renderer_matches_reference_and_backpropagates():
    torch.manual_seed(7)
    decoder = ChannelWiseGaussianSplatDecoder3D(
        n_channels=2,
        code_dim=4,
        grid_dim=16,
        latent_grid_dim=4,
        hidden_dim=8,
        gaussians_per_voxel=2,
        scale_min=0.35,
        scale_max=0.6,
        cutoff_sigma=2.0,
        opacity_threshold=0.0,
        query_chunk_size=5,
    )
    decoder.eval()
    codes = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
    query_points = torch.rand(1, 17, 3) * 2 - 1
    parameters = decoder.decode_parameters(codes)

    local = decoder._render_local(query_points, parameters)
    centers, scales, opacities = _flatten_parameters(parameters)
    reference = render_gaussians_chunked(
        query_points,
        centers,
        scales,
        opacities,
        query_chunk_size=4,
        gaussian_chunk_size=13,
        cutoff_sigma=decoder.cutoff_sigma,
    )

    torch.testing.assert_close(local, reference, atol=3e-6, rtol=3e-6)
    local.mean().backward()
    assert codes.grad is not None
    assert torch.isfinite(codes.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def test_decoded_parameters_respect_bounds():
    decoder = ChannelWiseGaussianSplatDecoder3D(
        n_channels=3,
        code_dim=5,
        grid_dim=20,
        latent_grid_dim=3,
        hidden_dim=7,
        gaussians_per_voxel=2,
        scale_min=0.25,
        scale_max=1.1,
        offset_bound=0.4,
        resolution=0.5,
    )
    codes = torch.randn(2, 5, 3, 3, 3) * 100
    parameters = decoder.decode_parameters(codes)
    anchors = decoder._latent_anchors(
        (3, 3, 3), device=codes.device, dtype=codes.dtype
    )
    spacing = 1.0
    max_offset = (
        parameters["centers"] - anchors[None, :, None, None, :]
    ).abs().max()
    scales_angstrom = parameters["scales"] * (
        decoder.resolution * decoder.grid_dim / 2.0
    )

    assert max_offset <= decoder.offset_bound * spacing + 1e-6
    assert scales_angstrom.min() >= decoder.scale_min - 1e-6
    assert scales_angstrom.max() <= decoder.scale_max + 1e-6
    assert parameters["opacities"].min() >= 0
    assert parameters["opacities"].max() <= 1


class DummyFieldMaker:
    @staticmethod
    def compute_voxel_grid(ligand, num_channels):
        return torch.zeros(ligand["xs"].shape[0], num_channels, 4, 4, 4)


class DummyEncoder(torch.nn.Module):
    def forward(self, voxels):
        batch = voxels.shape[0]
        return torch.arange(4 * 4 * 4 * 4, dtype=voxels.dtype).reshape(
            1, 4, 4, 4, 4
        ).expand(batch, -1, -1, -1, -1)


def _minimal_config(decoder_type="gaussian_splat"):
    return OmegaConf.create(
        {
            "decoder": {
                "type": decoder_type,
                "code_dim": 4,
                "hidden_dim": 8,
                "gaussians_per_voxel": 1,
                "scale_min": 0.35,
                "scale_max": 0.8,
                "offset_bound": 0.5,
                "opacity_threshold": 0.01,
                "initial_opacity": 0.05,
                "cutoff_sigma": 2.0,
                "query_chunk_size": 8,
                "gaussian_chunk_size": 8,
            },
            "encoder": {"downsample_map": [False, False, False]},
            "dset": {
                "elements": ["C", "N"],
                "n_channels": 2,
                "grid_dim": 8,
                "latent_grid_dim": 4,
                "resolution": 0.25,
            },
            "reg_weight": 0.0,
        }
    )


def test_factory_and_inference_keep_full_grid_only_for_gaussian_decoder():
    gaussian_config = _minimal_config()
    decoder = create_nf_decoder(gaussian_config, FabricStub())
    assert isinstance(decoder, ChannelWiseGaussianSplatDecoder3D)
    assert decoder.requires_full_latent_grid

    xs = torch.rand(2, 9, 3) * 1.8 - 0.9
    batch = {"ligand": {"xs": xs}}
    encoder = DummyEncoder()
    gaussian_codes, _ = infer_codes_batch(
        batch, encoder, DummyFieldMaker(), gaussian_config, xs=xs
    )
    assert gaussian_codes.shape == (2, 4, 4, 4, 4)
    assert decoder(xs, gaussian_codes).shape == (2, 9, 2)

    inr_config = _minimal_config(decoder_type="inr")
    inr_codes, _ = infer_codes_batch(
        batch, encoder, DummyFieldMaker(), inr_config, xs=xs
    )
    assert inr_codes.shape == (2, 9, 4)


def test_factory_rejects_unknown_decoder_type():
    config = _minimal_config(decoder_type="unknown")
    with pytest.raises(ValueError, match="Unknown decoder type"):
        create_nf_decoder(config, FabricStub())
