"""Opt-in VoxBind DDP entrypoint using hybrid Muon + auxiliary AdamW.

This file is deliberately separate from the active AdamW training entrypoint.
Running it is an explicit choice; importing it does not start training.

The underlying VoxBind trainer hard-codes construction of its AdamW class, so
this wrapper captures the model created by the trainer and replaces that
constructor with the VoxBind-aware optimizer builder.

Environment variables:
    VOXBIND_MUON_LR          hidden matrix/Conv3d learning rate (default 0.02)
    VOXBIND_MUON_MOMENTUM    momentum (default 0.95)
    VOXBIND_MUON_NS_STEPS    Newton--Schulz iterations (default 5)
    VOXBIND_MUON_AUX_LR      auxiliary AdamW LR (default: Hydra ``lr``)
    VOXBIND_MUON_AUX_EPS     auxiliary AdamW epsilon (default 1e-8)

Muon checkpoints can resume through the normal VoxBind path.  Loading an
existing AdamW optimizer checkpoint is rejected with a clear error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
VOXBIND_REPOSITORY = Path("/home1/irteam/VoxBind")
VOXBIND_SOURCE = VOXBIND_REPOSITORY / "voxbind"
for path in (
    PROJECT_ROOT,
    SCRIPTS_ROOT,
    VOXBIND_REPOSITORY,
    VOXBIND_SOURCE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ddp as upstream

from funcbind.models.muon import (
    MuonWithAuxAdamW,
    partition_voxbind_parameters,
)
from train_ddp_optimized_entry import FusedModelEma, ReducedSyncVoxelizer


_PARAMETER_NAMES: dict[int, str] = {}
_original_create_model = upstream.create_model


def _create_model_and_capture_names(*args, **kwargs):
    model = _original_create_model(*args, **kwargs)
    _PARAMETER_NAMES.clear()
    _PARAMETER_NAMES.update(
        (id(parameter), name) for name, parameter in model.named_parameters()
    )
    return model


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _make_voxbind_muon(params, lr=1e-5, weight_decay=1e-2, **_kwargs):
    parameters = list(params)
    named_parameters = [
        (_PARAMETER_NAMES.get(id(parameter), f"unnamed_{index}"), parameter)
        for index, parameter in enumerate(parameters)
    ]
    partition = partition_voxbind_parameters(named_parameters)

    muon_lr = _float_env("VOXBIND_MUON_LR", 0.02)
    aux_lr = _float_env("VOXBIND_MUON_AUX_LR", lr)
    momentum = _float_env("VOXBIND_MUON_MOMENTUM", 0.95)
    ns_steps = _int_env("VOXBIND_MUON_NS_STEPS", 5)
    aux_eps = _float_env("VOXBIND_MUON_AUX_EPS", 1e-8)

    groups = []
    if partition.muon:
        groups.append(
            {
                "params": [parameter for _, parameter in partition.muon],
                "use_muon": True,
                "lr": muon_lr,
                "momentum": momentum,
                "nesterov": True,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
                "group_name": "voxbind_hidden_weights",
            }
        )
    if partition.adamw:
        groups.append(
            {
                "params": [parameter for _, parameter in partition.adamw],
                "use_muon": False,
                "lr": aux_lr,
                "betas": (0.9, 0.95),
                "eps": aux_eps,
                "weight_decay": weight_decay,
                "group_name": "voxbind_aux_adamw",
            }
        )

    optimizer = MuonWithAuxAdamW(groups)
    print(
        "[muon-entry] optimizer prepared (not compatible with AdamW optimizer "
        "checkpoints): "
        f"Muon={len(partition.muon)} tensors/"
        f"{partition.muon_parameters / 1e6:.2f}M params at lr={muon_lr:g}; "
        f"aux-AdamW={len(partition.adamw)} tensors/"
        f"{partition.adamw_parameters / 1e6:.2f}M params at lr={aux_lr:g}; "
        f"frozen={partition.frozen_parameters / 1e6:.2f}M params",
        flush=True,
    )
    return optimizer


# Keep the throughput fixes, but optimizer selection remains exclusive to this
# entrypoint.  The already-running optimized AdamW process imports a different
# module and cannot be affected by these assignments.
upstream.create_model = _create_model_and_capture_names
upstream.AdamW = _make_voxbind_muon
upstream.ModelEma = FusedModelEma
upstream.Voxelizer = ReducedSyncVoxelizer


if __name__ == "__main__":
    print(
        "[muon-entry] hybrid Muon requested explicitly; fused EMA and "
        "two-chunk PyUUL voxelization enabled",
        flush=True,
    )
    upstream.main()
