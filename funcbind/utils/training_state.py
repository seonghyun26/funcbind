"""Bounded-memory checkpoint I/O for the DDP / ZeRO-1 training recipe.

All ranks must call save_training_state. Optimizer shards are written before the
atomic main checkpoint; that checkpoint is the commit record. Existing latest /
best checkpoints and their shards are never deleted here. Resume currently
requires the same world size and parameter partition and a shared filesystem.
"""
from pathlib import Path
import os
import random
import shutil
import uuid

import numpy as np
import torch


def load_cpu_checkpoint(path):
    # mmap avoids materializing the base checkpoint (including unused optimizer
    # and EMA tensors) independently in every worker. Never map it onto a GPU.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        raise RuntimeError("Convert this legacy checkpoint to torch.save's zip format "
                           "before using the bounded-memory loader") from exc


def unwrap_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def is_zero_optimizer(optimizer):
    from torch.distributed.optim import ZeroRedundancyOptimizer
    return isinstance(unwrap_optimizer(optimizer), ZeroRedundancyOptimizer)


def _local_optimizer(optimizer):
    optimizer = unwrap_optimizer(optimizer)
    return optimizer.optim if is_zero_optimizer(optimizer) else optimizer


def _parameter_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]]
            for group in _local_optimizer(optimizer).param_groups]


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _all_succeeded(fabric, error, operation):
    failed = fabric.all_reduce(torch.tensor(int(error is not None), device=fabric.device),
                               reduce_op="sum")
    if failed.item():
        raise RuntimeError(f"{operation} failed on {int(failed.item())} rank(s): "
                           f"{error or 'see the failing rank log'}")


def save_training_state(fabric, model, optimizer, ema, metadata, dirname,
                        is_best=False, reserve_gib=30):
    """Write unique optimizer/RNG shards and atomically publish a legacy-readable model."""
    root = Path(dirname)
    opt = unwrap_optimizer(optimizer)
    sharded = is_zero_optimizer(opt)
    local = dict(optimizer=_local_optimizer(opt).state_dict(),
                 parameter_names=_parameter_names(model, opt), rng=_rng_state())
    state = None
    if fabric.global_rank == 0:
        state = dict(metadata, state_dict=model.state_dict(), state_dict_ema=ema.state_dict())
    local_bytes = _tensor_bytes(local) + (_tensor_bytes(state) if state else 0)
    total_bytes = int(fabric.all_reduce(torch.tensor(local_bytes, device=fabric.device,
                                                    dtype=torch.int64), reduce_op="sum").item())
    error = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        needed = int(total_bytes * 1.02) + int(reserve_gib * 1024**3)
        if shutil.disk_usage(root).free < needed:
            raise OSError(f"Checkpoint needs {needed / 1024**3:.1f} GiB free including "
                          f"{reserve_gib} GiB reserve; no checkpoint files written")
    except Exception as exc:
        error = str(exc)
    _all_succeeded(fabric, error, "checkpoint space check")

    token = fabric.broadcast(uuid.uuid4().hex if fabric.global_rank == 0 else None, src=0)
    relative = f"optimizer_shards/{token}"
    shard_dir = root / relative
    error = None
    try:
        shard_dir.mkdir(parents=True, exist_ok=True)
        target = shard_dir / f"rank{fabric.global_rank:04d}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save(local, temporary)
        os.replace(temporary, target)
    except Exception as exc:
        error = str(exc)
        print(f"rank {fabric.global_rank}: checkpoint shard failed: {error}", flush=True)
    _all_succeeded(fabric, error, "optimizer shard save")

    checkpoint_path, best_path = root / "checkpoint.pth.tar", None
    error = None
    if fabric.global_rank == 0:
        try:
            state["optimizer"] = dict(format="funcbind-local-optimizer-v1",
                sharded=sharded, world_size=fabric.world_size, directory=relative,
                param_groups=[{k: v for k, v in g.items() if k != "params"}
                              for g in opt.param_groups])
            temporary = root / f"checkpoint.{token}.tmp"
            torch.save(state, temporary)
            os.replace(temporary, checkpoint_path)
            if is_best:
                best_path = root / "checkpoint_best.pth.tar"
                best_tmp = root / f"checkpoint_best.{token}.tmp"
                os.link(checkpoint_path, best_tmp)
                os.replace(best_tmp, best_path)
        except Exception as exc:
            error = str(exc)
    _all_succeeded(fabric, error, "main checkpoint save")
    return str(checkpoint_path), str(best_path) if best_path else None


def restore_optimizer_state(optimizer, state, model, fabric, checkpoint_dir):
    """Load only this worker's moments; reject incompatible partitions explicitly."""
    opt = unwrap_optimizer(optimizer)
    if state.get("format") != "funcbind-local-optimizer-v1":
        opt.load_state_dict(state)
        return
    if state["world_size"] != fabric.world_size or state["sharded"] != is_zero_optimizer(opt):
        raise ValueError("Sharded resume requires the saved world size and optimizer strategy")
    root = Path(checkpoint_dir).resolve()
    shard_dir = (root / state["directory"]).resolve()
    if not shard_dir.is_relative_to(root):
        raise ValueError("Optimizer shard path escapes checkpoint directory")
    saved = load_cpu_checkpoint(shard_dir / f"rank{fabric.global_rank:04d}.pt")
    if saved["parameter_names"] != _parameter_names(model, opt):
        raise ValueError("Optimizer parameter partition differs from saved checkpoint")
    if len(state["param_groups"]) != len(opt.param_groups):
        raise ValueError("Optimizer parameter group count differs from checkpoint")
    for group, saved_group in zip(opt.param_groups, state["param_groups"]):
        group.update(saved_group)
    _local_optimizer(opt).load_state_dict(saved["optimizer"])
    _restore_rng(saved["rng"])
