#!/usr/bin/env python
"""Small distributed AdamW/CPU-EMA/checkpoint regression, NOT an H100 capacity test."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from funcbind.models.phema import CPUPowerFunctionEMA, create_ema, ema_evaluation
from funcbind.utils.utils_fb import create_optimizer
from funcbind.utils.training_state import (
    load_cpu_checkpoint, restore_optimizer_state, save_training_state,
)


def inputs(rank, step, micro, device):
    x = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4) / 16
    x = x + rank * .3 + step * .1 + micro * .02
    return x, torch.sin(x)


def update(fabric, model, optimizer, step, reference=False):
    optimizer.zero_grad(set_to_none=False)
    for micro in range(2):
        ranks = range(fabric.world_size) if reference else [fabric.global_rank]
        for rank in ranks:
            x, y = inputs(rank, step, micro, fabric.device)
            if reference:
                with fabric.autocast():
                    loss = (model(x).float() - y).square().mean() / (2 * fabric.world_size)
                loss.backward()
            else:
                with fabric.no_backward_sync(model, enabled=micro == 0):
                    loss = (model(x).float() - y).square().mean() / 2
                    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)


def run(fabric, args):
    torch.set_num_threads(1)
    fabric.seed_everything(123)
    config = dict(lr=.01, wd=.0001, dset=dict(beta2=.95), precision="bf16-mixed",
                  ema_stds=[.05], performance=dict(optimizer_foreach=False,
                    optimizer_sharding="zero1", ema_cpu=True))
    with fabric.init_module():
        net = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Tanh(),
                                  torch.nn.Linear(4, 4, bias=False))
    reference = copy.deepcopy(net)
    reference_config = copy.deepcopy(config)
    reference_config["performance"]["optimizer_sharding"] = "none"
    refopt = create_optimizer(reference, reference_config, fabric)
    zero = create_optimizer(net, config, fabric)
    ema = create_ema(net, config, fabric)
    refema = CPUPowerFunctionEMA(reference, [.05]) if fabric.global_rank == 0 else None
    model, optimizer = fabric.setup(net, zero)
    max_diff = 0.
    for step in range(3):
        # Exercise changes to the wrapper LR and propagation to the local AdamW.
        for opt in (optimizer, refopt):
            opt.param_groups[0]["lr"] = .01 / (step + 1)
        update(fabric, model, optimizer, step)
        update(fabric, reference, refopt, step, reference=True)
        for p, q in zip(net.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
            assert p.dtype == torch.float32
            max_diff = max(max_diff, (p - q).detach().abs().max().item())
        if ema is not None:
            ema.update((step + 1) * 8 * fabric.world_size, 8 * fabric.world_size)
            refema.update((step + 1) * 8 * fabric.world_size, 8 * fabric.world_size)
            for name, value in ema.state_dict()["emas"][0].items():
                torch.testing.assert_close(value, refema.state_dict()["emas"][0][name], rtol=1e-5, atol=1e-6)
    counts = fabric.all_gather(torch.tensor(sum(s["exp_avg"].numel()
        for s in zero.optim.state.values()), device=fabric.device)).cpu().tolist()
    assert sum(counts) == sum(p.numel() for p in net.parameters())
    assert max(counts) < sum(counts)
    for state in zero.optim.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    if ema is not None:
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        with ema_evaluation(ema) as averaged, fabric.autocast(), torch.no_grad():
            assert torch.isfinite(averaged(torch.ones(4, 4, device=fabric.device))).all()
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    fabric.barrier()
    metadata = dict(config=config, epoch=1, global_step=3, acc_iter=24 * fabric.world_size,
                    best_res=1., code_stats={})
    checkpoint, _ = save_training_state(fabric, model, optimizer, ema, metadata,
                                        args.out, is_best=True, reserve_gib=30)
    saved = load_cpu_checkpoint(checkpoint)
    resumed = copy.deepcopy(net)
    resumed.load_state_dict(saved["state_dict"])
    resumed_zero = create_optimizer(resumed, config, fabric)
    resumed_model, resumed_opt = fabric.setup(resumed, resumed_zero)
    restore_optimizer_state(resumed_opt, saved["optimizer"], resumed_model, fabric, args.out)
    update(fabric, model, optimizer, 3)
    update(fabric, resumed_model, resumed_opt, 3)
    for p, q in zip(net.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # Reject low disk before creating a new shard/main checkpoint.
    try:
        save_training_state(fabric, model, optimizer, ema, metadata, args.out,
                            reserve_gib=10**9)
    except RuntimeError as exc:
        assert "space check" in str(exc)
    else:
        raise AssertionError("disk guard did not reject the save")
    fabric.barrier()
    if fabric.global_rank == 0:
        report = dict(status="passed", accelerator=args.accelerator, devices=fabric.world_size,
            precision="bf16-mixed", reference_max_abs_diff=max_diff,
            resume_max_abs_diff=0., moment_numel_per_rank=counts,
            cpu_ema=True, checkpoint_roundtrip=True, disk_guard=True,
            full_model_capacity_validated=False)
        Path(args.out, "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.devices < 2:
        parser.error("at least two devices/processes are required")
    strategy = DDPStrategy(process_group_backend="gloo" if args.accelerator == "cpu" else "nccl",
        start_method="fork" if args.accelerator == "cpu" else "popen", gradient_as_bucket_view=True)
    fabric = Fabric(accelerator=args.accelerator, devices=args.devices,
                    precision="bf16-mixed", strategy=strategy)
    fabric.launch(run, args)


if __name__ == "__main__":
    main()
