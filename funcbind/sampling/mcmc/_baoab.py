import dataclasses
from dataclasses import dataclass

import math
from typing import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ._utils import initialize_velocity, create_score_fn


@dataclass
class BAOAB:
    delta: float = 1.0
    sigma: float = 5.0
    friction: float = 1.0
    M: float = 1.0
    steps: int = 128
    save_trajectory: bool = False
    save_every_n_steps: int = 1
    burn_in_steps: int = 0
    verbose: bool = False
    cpu_offload: bool = False
    v_init: str | Tensor = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: float | None = None

    def __post_init__(self):
        if isinstance(self.v_init, str):
            if self.v_init not in {"gaussian", "zero"}:
                raise RuntimeError(f"{self.v_init} not in (gaussian, zero)")

    def __call__(self, y: torch.Tensor, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return baoab(y, score_fn, **kwargs)


def baoab(
    y: torch.Tensor,
    score_fn: Callable,
    steps: int,
    v_init: str | Tensor = "zero",
    save_trajectory=False,
    save_every_n_steps=1,
    burn_in_steps=0,
    verbose=False,
    cpu_offload=False,
    delta: float = 1.0,
    sigma: float = 5.0,
    friction: float = 1.0,
    M: float = 1.0,
    inverse_temperature: float = 1.0,
    score_fn_clip: float | None = None,
    **_,
):
    """
    Note
    ----
    BAOAB splitting scheme.

    See section 7.3 of "Molecular Dynamics: With Deterministic and Stochastic Numerical Methods" (page 271)
    """
    i = 0
    delta *= sigma
    y_traj = [] if save_trajectory else None
    if y_traj is not None and i >= burn_in_steps:
        y_traj.append(y.detach().cpu() if cpu_offload else y.detach())

    u = pow(M, -1)  # inverse mass
    zeta2 = math.sqrt(1 - math.exp(-2 * friction))

    v_init = initialize_velocity(v_init=v_init, y=y, u=u)
    v = v_init

    # step zero is initialization
    steps_iter = range(1, steps)
    if verbose:
        steps_iter = tqdm(steps_iter, leave=False, desc="BAOAB")

    # Wrap score function with clipping and scaling by inverse temperature.
    score_fn_processed = create_score_fn(score_fn, inverse_temperature, score_fn_clip)
    psi, orig_score = score_fn_processed(y)
    score_traj = [orig_score.detach().cpu() if cpu_offload else orig_score.detach()]

    for i in steps_iter:
        v = v + u * (delta / 2) * psi  # v_{t+1/2}
        y = y + (delta / 2) * v  # y_{t+1/2}

        R = torch.randn_like(y)
        vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R
        y = y + (delta / 2) * vhat  # y_{t+1}

        psi, orig_score = score_fn_processed(y)
        v = vhat + (delta / 2) * psi  # v_{t+1}

        if (
            y_traj is not None
            and ((i % save_every_n_steps) == 0)
            and (i >= burn_in_steps)
        ):
            y_traj.append(y.detach().cpu() if cpu_offload else y.detach())
            score_traj.append(
                orig_score.detach().cpu() if cpu_offload else orig_score.detach()
            )

    if y_traj is not None:
        y_traj = torch.stack(y_traj)

    if len(score_traj) > 0:
        score_traj = torch.stack(score_traj)

    return y, v, y_traj, score_traj
