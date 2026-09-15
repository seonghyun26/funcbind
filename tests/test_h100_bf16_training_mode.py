from pathlib import Path

import torch
from lightning import Fabric
from hydra import compose, initialize_config_dir

from funcbind.models.adamw import AdamW
from funcbind.models.phema import PowerFunctionEMA


REPO = Path(__file__).resolve().parents[1]


def test_h100_config_preserves_mixed_precision(monkeypatch):
    # A stale environment value from the former true-BF16 profile must not
    # silently change the precision requested for this experiment.
    monkeypatch.setenv("FUNCBIND_PRECISION", "bf16-true")
    with initialize_config_dir(
        config_dir=str(REPO / "funcbind/configs"), version_base=None
    ):
        config = compose(config_name="train_fb_mcpp_holo_density_h100")

    assert config.precision == "bf16-mixed"
    assert config.dset.batch_size == 1
    assert config.dset.val_batch_size == 1
    assert config.accum_steps == 95
    assert config.performance.compile_backend == ""
    assert not config.performance.optimizer_foreach
    assert config.performance.optimizer_sharding == "zero1"
    assert config.performance.ema_cpu
    assert config.performance.activation_checkpointing
    assert not config.performance.zero_grad_set_to_none
    assert config.performance.validation_render_multiplier == 1
    assert config.dset.num_workers == 2
    assert config.dset.prefetch_factor == 2
    assert config.sampling.batch_size_render == 256
    assert config.sampling.batch_size_render_codes == 1


def test_mixed_bf16_adamw_and_ema_keep_fp32_state():
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    model = torch.nn.Linear(4, 4, bias=False)
    ema = PowerFunctionEMA(model, stds=[0.05], foreach=True)
    optimizer = AdamW(
        model.parameters(), lr=0.1, betas=(0.9, 0.95), foreach=False
    )
    output_dtypes = []
    hook = model.register_forward_hook(
        lambda module, args, output: output_dtypes.append(output.dtype)
    )
    wrapped_model, wrapped_optimizer = fabric.setup(model, optimizer)
    ema.to(device=model.weight.device, dtype=model.weight.dtype)
    prediction = wrapped_model(torch.ones(8, 4))
    fabric.backward(prediction.float().sum())
    wrapped_optimizer.step()
    ema.update(cur_nimg=8, batch_size=8)
    hook.remove()

    assert output_dtypes == [torch.bfloat16]
    assert model.weight.dtype == torch.float32
    assert model.weight.grad.dtype == torch.float32
    assert next(ema.emas[0].parameters()).dtype == torch.float32
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[model.weight]["exp_avg_sq"].dtype == torch.float32
    assert torch.isfinite(model.weight).all()
    assert torch.isfinite(next(ema.emas[0].parameters())).all()
