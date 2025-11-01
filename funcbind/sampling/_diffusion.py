from typing import Callable

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm


def sigma_grid(
    sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, N: int = 64
) -> Tensor:
    step_indices = torch.arange(N)
    t_steps = (sigma_max ** (1 / rho) + (step_indices / (N - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps


def heun_step(f: Callable, y_hat, t_hat, t_next, corrector: bool = True):
    # Euler step
    d_cur = f(y_hat, t_hat)  # -t_cur * score_fn(y_cur, t_cur)
    y_next = y_hat + (t_next - t_hat) * d_cur
    if corrector:
        # Second order corrector
        d_prime = f(y_next, t_next)  # -t_next * score_fn(y_next, t_next)
        y_next = y_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return y_next


def integrate_heun(
    f: Callable, y_next, t_steps, save_trajectory: bool = False, verbose: bool = False,
    S_churn=0, S_min=0.05, S_max=50, S_noise=1.003, num_steps: int = 64, second_order: bool = True,
):
    traj = [] if save_trajectory else None
    if save_trajectory:
        traj.append(y_next)

    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), desc="heun", disable=not verbose
    ):
        y_cur = y_next
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            y_hat = y_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(y_cur)
        else:
            t_hat = t_cur
            y_hat = y_cur
        y_next = heun_step(f, y_hat, t_hat, t_next, corrector=(i < num_steps - 1) and second_order)

        if save_trajectory:
            traj.append(y_next)

    if save_trajectory:
        traj = torch.stack(traj, dim=0)

    return y_next, traj


class DiffusionSampler:
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        N: int = 64,
        y_init_distribution: torch.distributions.Distribution | None = None,
        verbose: bool = False,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
        save_trajectory=False,
        beta=1.0,
        **kwargs,
    ):
        self.t_steps = sigma_grid(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, N=N)
        self.sigma = sigma_max  # compat
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.N = N
        self.y_init_distribution = y_init_distribution
        self.verbose = verbose
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.beta = beta
        self.save_trajectory = save_trajectory

    def sample(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
    ):
        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError("either y_init and y_init_distribution must be supplied")
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,))

        def f(y, t):
            Dy = score_fn(y, t) * self.beta
            return -t * Dy

        S_noise = self.S_noise
        y, y_traj = integrate_heun(f, y_init, self.t_steps, save_trajectory=self.save_trajectory, verbose=self.verbose, S_churn=self.S_churn, S_min=self.S_min, S_max=self.S_max, S_noise=S_noise, num_steps=self.N)

        return {"sample": y.to("cpu") if y is not None else None, "xhat_traj": y_traj.to("cpu") if y_traj is not None else None}
