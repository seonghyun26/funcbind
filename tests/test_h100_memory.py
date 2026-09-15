import copy

import pytest
import torch

from funcbind.models.phema import CPUPowerFunctionEMA, PowerFunctionEMA, create_ema
from funcbind.models.unet3d import Block, BlockUncond, enable_activation_checkpointing


def test_cpu_ema_parity_and_exception_safe_validation():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.1), torch.nn.Linear(8, 4))
    model[1].eval()
    model.register_buffer("counter", torch.tensor(0))
    ema = CPUPowerFunctionEMA(model, stds=[.05], chunk_numel=3)
    reference = PowerFunctionEMA(model, stds=[.05], foreach=False)
    for step in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.1 * (step + 1))
            model.counter.add_(1)
        ema.update((step + 1) * 16, 16)
        reference.update((step + 1) * 16, 16)
    expected = reference.get()[0][0].state_dict()
    for name, value in ema.state_dict()["emas"][0].items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert value.device.type == "cpu"
    original = copy.deepcopy(model.state_dict())
    identities = [id(p) for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError, match="validation failure"):
        with ema.average_parameters() as averaged:
            assert averaged is model
            for name, value in averaged.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            raise RuntimeError("validation failure")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert [id(p) for p in model.parameters()] == identities
    assert [m.training for m in model.modules()] == modes


@pytest.mark.parametrize("conditional", [False, True])
def test_checkpoint_recompute_preserves_weights_gradients_and_rng(conditional):
    torch.manual_seed(7)
    block = (Block(4, 4, 4, dropout=.2) if conditional else BlockUncond(4, 4, dropout=.2))
    if conditional:
        with torch.no_grad():
            block.cond_gain.fill_(.3)
    checked = copy.deepcopy(block)
    enable_activation_checkpointing(checked)
    for step in range(3):
        # Internal blocks receive BF16 activations from the preceding MPConv.
        inputs = [torch.randn(1, 4, 4, 4, 4, dtype=torch.bfloat16, requires_grad=True)]
        if conditional:
            inputs.append(torch.randn(1, 4, 4, 4, 4, dtype=torch.bfloat16, requires_grad=True))
        other_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        torch.manual_seed(11 + step)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = block(*inputs)
        y.float().square().mean().backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(11 + step)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            z = checked(*other_inputs)
        z.float().square().mean().backward()
        assert torch.equal(rng, torch.get_rng_state())
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        for x, other in zip(inputs, other_inputs):
            torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
        with torch.no_grad():
            for p, q in zip(block.parameters(), checked.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                p.add_(p.grad, alpha=-.01)
                q.add_(q.grad, alpha=-.01)
        block.zero_grad(set_to_none=True)
        checked.zero_grad(set_to_none=True)


def test_nonzero_rank_has_no_cpu_ema():
    class RankOne:
        global_rank = 1
    assert create_ema(torch.nn.Linear(2, 2), dict(ema_stds=[.05], precision="bf16-mixed",
        performance=dict(ema_cpu=True)), RankOne()) is None


def test_frozen_inference_preserves_autocast_without_ddp():
    from lightning import Fabric
    from funcbind.utils.utils_nf import FrozenInferenceModule
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    module = torch.nn.Linear(4, 4)
    observed = []
    module.register_forward_hook(lambda m, args, out: observed.append(out.dtype))
    wrapped = FrozenInferenceModule(module, fabric)
    output = wrapped(torch.ones(2, 4))
    assert observed == [torch.bfloat16]
    assert output.dtype == torch.float32
    assert not output.requires_grad
    assert all(not p.requires_grad for p in wrapped.parameters())
    assert wrapped.in_features == 4


def test_memory_estimate_distinguishes_shards_and_cpu_ema():
    from scripts.preflight_mcp_density import estimate_memory, DEFAULT_PARAMS
    config = dict(precision="bf16-mixed", performance={})
    baseline = sum(estimate_memory(config, DEFAULT_PARAMS, 8).values())
    config["performance"] = dict(optimizer_sharding="zero1", ema_cpu=True)
    sharded = sum(estimate_memory(config, DEFAULT_PARAMS, 8).values())
    assert baseline == pytest.approx(95.76324, abs=.001)
    assert sharded == pytest.approx(43.09346, abs=.001)
    assert sum(estimate_memory(config, DEFAULT_PARAMS, 1).values()) > 76


def test_non_checkpointed_mpconv_still_compiles():
    from funcbind.models.unet3d import MPConv
    torch.manual_seed(3)
    original = MPConv(4, 4, kernel=[])
    compiled = torch.compile(copy.deepcopy(original), backend="eager", fullgraph=True)
    x = torch.randn(2, 4)
    torch.testing.assert_close(compiled(x), original(x), rtol=0, atol=0)
