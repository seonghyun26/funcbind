import copy

import pytest
import torch

from funcbind.models.muon import (
    MuonWithAuxAdamW,
    build_voxbind_muon_optimizer,
    muon_update,
    partition_voxbind_parameters,
    zeropower_via_newtonschulz5,
)


class TinyVoxBind(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_encoder = torch.nn.Conv3d(7, 4, kernel_size=1)
        self.hidden = torch.nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.norm = torch.nn.GroupNorm(2, 8)
        self.density_encoder = torch.nn.Conv3d(1, 4, kernel_size=1)
        self.final_ligand = torch.nn.Conv3d(8, 7, kernel_size=1)
        for parameter in self.density_encoder.parameters():
            parameter.requires_grad = False


def test_newton_schulz_is_shape_preserving_and_finite():
    matrix = torch.randn(7, 11)
    result = zeropower_via_newtonschulz5(matrix, steps=5)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


def test_muon_flattens_voxbind_conv3d_weights():
    grad = torch.randn(8, 4, 3, 3, 3)
    momentum = torch.zeros_like(grad)
    result = muon_update(grad, momentum)
    assert result.shape == grad.shape
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(momentum) > 0


def test_voxbind_partition_uses_muon_only_for_hidden_matrix_weights():
    model = TinyVoxBind()
    partition = partition_voxbind_parameters(model.named_parameters())
    muon_names = {name for name, _ in partition.muon}
    adamw_names = {name for name, _ in partition.adamw}
    frozen_names = {name for name, _ in partition.frozen}

    assert muon_names == {"hidden.weight"}
    assert "hidden.bias" in adamw_names
    assert "norm.weight" in adamw_names
    assert "ligand_encoder.weight" in adamw_names
    assert "final_ligand.weight" in adamw_names
    assert frozen_names == {
        "density_encoder.weight",
        "density_encoder.bias",
    }


def test_hybrid_optimizer_steps_and_round_trips_state():
    torch.manual_seed(7)
    model = TinyVoxBind()
    optimizer, partition = build_voxbind_muon_optimizer(
        model,
        muon_lr=0.02,
        adamw_lr=1e-3,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name])
        else:
            assert torch.equal(parameter, before[name])
        assert torch.isfinite(parameter).all()

    state_dict = copy.deepcopy(optimizer.state_dict())
    restored, restored_partition = build_voxbind_muon_optimizer(model)
    restored.load_state_dict(state_dict)
    assert len(restored.state) == len(optimizer.state)
    assert restored_partition.muon_parameters == partition.muon_parameters


def test_muon_rejects_plain_adamw_checkpoint():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    muon = MuonWithAuxAdamW(
        [{"params": [parameter], "use_muon": True}]
    )
    adamw = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="does not contain Muon"):
        muon.load_state_dict(adamw.state_dict())


def test_zero_gradients_do_not_create_nans():
    matrix = torch.nn.Parameter(torch.ones(3, 5))
    vector = torch.nn.Parameter(torch.ones(5))
    optimizer = MuonWithAuxAdamW(
        [
            {"params": [matrix], "use_muon": True},
            {"params": [vector], "use_muon": False},
        ]
    )
    matrix.grad = torch.zeros_like(matrix)
    vector.grad = torch.zeros_like(vector)
    optimizer.step()
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()
