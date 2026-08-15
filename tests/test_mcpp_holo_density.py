import json

import numpy as np
import pytest
import torch

from funcbind.dataset.mcpp_holo_density import (
    MCPPHoloDensityStore,
    mcpp_target_id,
    resample_holo_box,
)
from funcbind.models.density_condition import (
    DensityCondition,
    apply_density_residual,
)


def test_target_id_comes_from_receptor_directory():
    assert mcpp_target_id("dataset/data/mcpp_dataset/6xif/6xif_12_mut3-MCP.pdb") == "6xif"
    assert mcpp_target_id("/tmp/mcpp_dataset/1abc/1abc-CP.sdf") == "1abc"


def test_canonical_crop_and_augmentation_are_coaligned():
    g_box = 80
    box = np.arange(g_box ** 3, dtype=np.float32).reshape(g_box, g_box, g_box)
    crop = resample_holo_box(box, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(crop, box[8:72, 8:72, 8:72])

    gaussian = np.zeros((g_box,) * 3, dtype=np.float32)
    gaussian[g_box // 2, g_box // 2, g_box // 2] = 1.0
    shifted = resample_holo_box(
        gaussian,
        np.zeros(3, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        translation=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.unravel_index(np.argmax(shifted), shifted.shape) == (36, 32, 32)


def test_out_of_cache_crop_is_unavailable_instead_of_zero_padded():
    box = np.zeros((80,) * 3, dtype=np.float32)
    assert resample_holo_box(box, np.array([20.0, 0.0, 0.0])) is None


def test_store_reads_target_box_and_applies_recipe(tmp_path):
    g_box = 80
    boxes = np.memmap(
        tmp_path / "boxes_float16.dat",
        dtype=np.float16,
        mode="w+",
        shape=(1, g_box, g_box, g_box),
    )
    boxes[0] = 0.5
    boxes.flush()
    del boxes
    (tmp_path / "meta.json").write_text(json.dumps({
        "g_box": g_box,
        "resolution": 0.25,
        "n_boxes": 1,
        "dtype": "float16",
        "box_file": "boxes_float16.dat",
        "normalization": {
            "scheme": "arcsinh_zscore",
            "arcsinh_scale": 0.5,
            "mu_a": 0.0,
            "sigma_a": 1.0,
        },
    }))
    (tmp_path / "manifest.json").write_text(json.dumps([{
        "target_id": "6xif",
        "box_index": 0,
        "reference_center": [0.0, 0.0, 0.0],
        "reference_ligand": "6xif-CP.sdf",
    }]))

    store = MCPPHoloDensityStore(tmp_path)
    density, available = store.load("6xif", torch.zeros(3))
    assert bool(available)
    assert density.shape == (64, 64, 64)
    torch.testing.assert_close(density, torch.full_like(density, np.arcsinh(1.0)))
    missing, missing_available = store.load("xxxx", torch.zeros(3))
    assert not bool(missing_available)
    assert torch.count_nonzero(missing) == 0


def test_zero_init_has_exact_parity_but_projection_gets_gradient(monkeypatch):
    class FakeEncoder(torch.nn.Module):
        def forward_features(self, x):
            base = x.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)
            features = torch.arange(32, dtype=x.dtype).reshape(1, 8, 4)
            return base + features

        def _pool_groups(self, tokens):
            return tokens

    monkeypatch.setattr(
        "funcbind.models.density_condition.build_density_encoder",
        lambda *_args, **_kwargs: FakeEncoder(),
    )
    condition = DensityCondition(
        density_cfg={"dim": 4},
        code_dim=3,
        code_grid_dim=2,
        voxbind_root=".",
        hidden=5,
        amp=False,
    )
    density_input = torch.randn(2, 13, 4, 4, 4)
    delta = condition(density_input)
    assert torch.count_nonzero(delta) == 0
    delta.sum().backward()
    assert condition.proj[-1].weight.grad.abs().sum() > 0


def test_missing_map_mask_is_exact_noop_after_projection_learns_bias():
    receptor = torch.randn(2, 3, 2, 2, 2)
    learned_delta = torch.ones_like(receptor)
    fused = apply_density_residual(
        receptor,
        learned_delta,
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(fused[0], receptor[0] + 1)
    torch.testing.assert_close(fused[1], receptor[1])


def test_kabsch_recovers_mcp_to_deposited_frame():
    pytest.importorskip("gemmi")
    from funcbind.dataset.prepare_mcpp_holo_density import kabsch

    rng = np.random.default_rng(4)
    local = rng.normal(size=(40, 3))
    q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    translation = np.array([4.0, -2.0, 1.5])
    deposited = local @ q.T + translation
    recovered_r, recovered_t, rmsd = kabsch(local, deposited)
    np.testing.assert_allclose(recovered_r, q, atol=1e-10)
    np.testing.assert_allclose(recovered_t, translation, atol=1e-10)
    assert rmsd < 1e-10
