"""Muon optimizer support for VoxBind-style 3D convolutional models.

The implementation follows the reference Muon algorithm:

* Nesterov momentum for hidden matrix-shaped parameters.
* Newton--Schulz orthogonalization of the momentum update.
* Auxiliary AdamW for parameters that should not use Muon.

VoxBind is trained with ordinary replicated DDP, which already all-reduces the
gradients.  Every rank therefore performs the same local optimizer update.  We
intentionally do not use the parameter-sharded/all-gather Muon variant: a
rank-0-only VoxBind checkpoint would otherwise contain only a shard of the
optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(matrix: Tensor, steps: int = 5) -> Tensor:
    """Approximate ``U @ V.T`` for ``matrix = U @ S @ V.T``.

    Muon's quintic Newton--Schulz iteration is stable in bfloat16 and avoids an
    explicit SVD.  Batched matrices are supported, although VoxBind currently
    passes one flattened convolutional kernel at a time.
    """

    if matrix.ndim < 2:
        raise ValueError(
            "Newton--Schulz orthogonalization requires a matrix-shaped tensor"
        )
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.to(dtype=torch.bfloat16)
    transposed = matrix.shape[-2] > matrix.shape[-1]
    if transposed:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x

    if transposed:
        x = x.mT
    return x


@torch.no_grad()
def muon_update(
    grad: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> Tensor:
    """Return one orthogonalized Muon update for a matrix or convolution."""

    if grad.ndim < 2:
        raise ValueError("Muon is only defined for matrix-shaped parameters")

    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = torch.lerp(grad, momentum_buffer, momentum) if nesterov else momentum_buffer

    original_shape = update.shape
    if update.ndim > 2:
        # Conv1d/2d/3d: output channels form rows; all remaining axes form
        # columns.  The reference implementation does this for Conv2d; VoxBind
        # needs the same operation for its Conv3d/ConvTranspose3d weights.
        update = update.reshape(original_shape[0], -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * max(1.0, update.shape[-2] / update.shape[-1]) ** 0.5
    return update.reshape(original_shape)


class MuonWithAuxAdamW(Optimizer):
    """Hybrid Muon/AdamW optimizer with checkpoint-compatible parameter groups.

    Every parameter group must contain ``use_muon``.  Muon groups accept
    ``lr``, ``momentum``, ``nesterov``, ``ns_steps``, and ``weight_decay``.
    Auxiliary AdamW groups accept ``lr``, ``betas``, ``eps``, and
    ``weight_decay``.

    This optimizer is suitable for replicated single-device training and DDP.
    Gradients with ``None`` values are skipped, matching PyTorch optimizer
    semantics and VoxBind's ``find_unused_parameters=True`` training setup.
    """

    def __init__(self, param_groups: Sequence[dict]):
        if not param_groups:
            raise ValueError("at least one parameter group is required")

        normalized_groups: list[dict] = []
        for raw_group in param_groups:
            if "use_muon" not in raw_group:
                raise ValueError("every parameter group must define use_muon")

            group = dict(raw_group)
            group["params"] = list(group["params"])
            if not group["params"]:
                continue

            if group["use_muon"]:
                incompatible = [p.shape for p in group["params"] if p.ndim < 2]
                if incompatible:
                    raise ValueError(
                        "Muon groups may only contain matrix-shaped parameters; "
                        f"found {incompatible}"
                    )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)

            self._validate_group(group)
            normalized_groups.append(group)

        if not normalized_groups:
            raise ValueError("all parameter groups were empty")
        super().__init__(normalized_groups, defaults={})

    @staticmethod
    def _validate_group(group: dict) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if lr < 0:
            raise ValueError(f"learning rate must be non-negative, got {lr}")
        if weight_decay < 0:
            raise ValueError(
                f"weight decay must be non-negative, got {weight_decay}"
            )

        if group["use_muon"]:
            momentum = float(group["momentum"])
            if not 0 <= momentum < 1:
                raise ValueError(f"momentum must be in [0, 1), got {momentum}")
            if int(group["ns_steps"]) < 1:
                raise ValueError("ns_steps must be positive")
        else:
            beta1, beta2 = group["betas"]
            if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
                raise ValueError(f"AdamW betas must be in [0, 1), got {group['betas']}")
            if float(group["eps"]) < 0:
                raise ValueError(f"AdamW eps must be non-negative, got {group['eps']}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict) -> None:
        for parameter in group["params"]:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )

            update = muon_update(
                grad,
                state["momentum_buffer"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_steps=group["ns_steps"],
            )
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-group["lr"])

    def _step_adamw_group(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("auxiliary AdamW does not support sparse gradients")

            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denominator = exp_avg_sq.sqrt() / bias_correction2**0.5
            denominator.add_(group["eps"])

            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.addcdiv_(
                exp_avg,
                denominator,
                value=-group["lr"] / bias_correction1,
            )

    def load_state_dict(self, state_dict: dict) -> None:
        """Reject an AdamW checkpoint instead of silently misreading its state."""

        checkpoint_groups = state_dict.get("param_groups", [])
        if not checkpoint_groups or not any(
            group.get("use_muon", False) for group in checkpoint_groups
        ):
            raise ValueError(
                "The checkpoint does not contain Muon parameter groups. "
                "Do not resume a Muon run from an AdamW optimizer state; start a "
                "new run or load model weights without optimizer state."
            )
        super().load_state_dict(state_dict)


@dataclass(frozen=True)
class VoxBindParameterPartition:
    """Named trainable parameter split used by the VoxBind Muon builder."""

    muon: tuple[tuple[str, Parameter], ...]
    adamw: tuple[tuple[str, Parameter], ...]
    frozen: tuple[tuple[str, Parameter], ...]

    @property
    def muon_parameters(self) -> int:
        return sum(parameter.numel() for _, parameter in self.muon)

    @property
    def adamw_parameters(self) -> int:
        return sum(parameter.numel() for _, parameter in self.adamw)

    @property
    def frozen_parameters(self) -> int:
        return sum(parameter.numel() for _, parameter in self.frozen)


_VOXBIND_AUX_PREFIXES = (
    # These are the raw ligand/pocket input stems.  Muon's reference guidance
    # keeps the first convolution of a ConvNet on AdamW.
    "ligand_encoder.",
    "pocket_encoder.",
    # A pretrained density encoder should remain frozen for the requested
    # experiment.  If it is deliberately unfrozen later, keep it on AdamW by
    # default rather than applying Muon during pretrained fine-tuning.
    "density_encoder.",
    # The generated ligand grid is the model output head.
    "final_ligand.",
)


def partition_voxbind_parameters(
    named_parameters: Iterable[tuple[str, Parameter]],
) -> VoxBindParameterPartition:
    """Split VoxBind parameters into hidden Muon, auxiliary AdamW, and frozen."""

    muon: list[tuple[str, Parameter]] = []
    adamw: list[tuple[str, Parameter]] = []
    frozen: list[tuple[str, Parameter]] = []
    seen: set[int] = set()

    for name, parameter in named_parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)

        if not parameter.requires_grad:
            frozen.append((name, parameter))
            continue

        use_aux_adamw = parameter.ndim < 2 or name.startswith(
            _VOXBIND_AUX_PREFIXES
        )
        if use_aux_adamw:
            adamw.append((name, parameter))
        else:
            muon.append((name, parameter))

    return VoxBindParameterPartition(
        muon=tuple(muon),
        adamw=tuple(adamw),
        frozen=tuple(frozen),
    )


def build_voxbind_muon_optimizer(
    model: torch.nn.Module,
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-5,
    weight_decay: float = 1e-2,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> tuple[MuonWithAuxAdamW, VoxBindParameterPartition]:
    """Build the intended hybrid optimizer and return its parameter summary."""

    partition = partition_voxbind_parameters(model.named_parameters())
    groups = []
    if partition.muon:
        groups.append(
            {
                "params": [parameter for _, parameter in partition.muon],
                "use_muon": True,
                "lr": muon_lr,
                "momentum": momentum,
                "nesterov": nesterov,
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
                "lr": adamw_lr,
                "betas": adamw_betas,
                "eps": adamw_eps,
                "weight_decay": weight_decay,
                "group_name": "voxbind_aux_adamw",
            }
        )

    optimizer = MuonWithAuxAdamW(groups)
    return optimizer, partition
