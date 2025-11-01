import math
from typing import Callable, Optional

import torch
from torch import Tensor
from lightning.pytorch.utilities import rank_zero_only
from tqdm.auto import tqdm


class MeanConditionedScoreModel:
    def __init__(self, score_fn, sigma: float, y_bar_prev: torch.Tensor, t: int):
        self.score_fn = score_fn
        self.sigma = sigma
        self.y_bar_prev = y_bar_prev
        self.t = t

    def __call__(self, y_t):
        return self.score(y_t)

    def score(self, y_t):
        # see eq. 4.9
        y_bar_cur = self.y_bar_prev + (y_t - self.y_bar_prev) / self.t
        sigma_t = self.sigma / math.sqrt(self.t)
        term1 = self.score_fn(y_bar_cur, sigma=sigma_t) / self.t
        term2 = (y_bar_cur - y_t) / self.sigma**2.0
        return term1 + term2


def xhat_fn(score_fn, y, sigma, beta=1.0):
    s = score_fn(y, sigma)
    return y + (sigma**2) * s * beta


class MultiMeasurementOATSampler:
    def __init__(
        self,
        mcmc: Callable,
        sigma: float = 1.0,
        m: int = 4,
        beta: float = 1.0,
        warm_start: bool = False,
        callbacks: Optional[dict[str, Callable]] = None,
        y_init_distribution: Optional[torch.distributions.Distribution] = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.mcmc = mcmc
        self.sigma = float(sigma)
        self.m = m
        self.beta = beta
        self.warm_start = warm_start
        self.callbacks = callbacks
        self.y_init_distribution = y_init_distribution
        self.verbose = verbose and rank_zero_only.rank == 0

    def walk(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
    ):
        y_traj = None
        t_traj = None

        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError(
                    "either y_init and y_init_distribution must be supplied"
                )
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,))

        y_t = torch.zeros_like(y_init)
        y_bar_prev = torch.zeros_like(y_init)

        t_iter = range(1, self.m + 1)
        if self.verbose:
            t_iter = tqdm(t_iter, leave=False, desc="measurement")

        for t in t_iter:
            if t > 1:
                if self.warm_start:
                    xhat_prev = xhat_fn(
                        score_fn, y_bar_prev, sigma=self.sigma / math.sqrt(t - 1), beta=self.beta
                    )
                    y_init = xhat_prev + self.sigma * torch.randn_like(xhat_prev)
                else:
                    # cold start
                    if self.y_init_distribution is None:
                        raise RuntimeError(
                            "y_init_distribution must be supplied for cold starts"
                        )
                    y_init = self.y_init_distribution.sample(sample_shape=(batch_size,))

            if self.callbacks:
                for _, c in self.callbacks.items():
                    c.on_before_sample(self.mcmc, t=t)

            mean_conditioned_score_fn = MeanConditionedScoreModel(
                score_fn=score_fn, sigma=self.sigma, y_bar_prev=y_bar_prev, t=t
            )
            y_t, _, y_traj_t, _ = self.mcmc(y_init, mean_conditioned_score_fn)

            if self.callbacks:
                for _, c in self.callbacks.items():
                    c.on_after_sample(self.mcmc, t=t)

            if y_traj_t is not None:
                if y_traj is None:
                    y_traj = []
                    t_traj = []

                y_bar_traj_t = (
                    y_bar_prev.to(y_traj_t.device)
                    + (y_traj_t - y_bar_prev.to(y_traj_t.device)) / t
                )

                y_traj.append(y_bar_traj_t.detach())
                t_traj.append(
                    t * torch.ones(y_traj_t.size(0), device=y_traj_t.device, dtype=int)
                )

            y_bar_prev = y_bar_prev + (y_t - y_bar_prev) / t

        if y_traj is not None:
            y_traj = torch.cat(y_traj, dim=0)
            t_traj = torch.cat(t_traj, dim=0)

        return {"y_bar": y_bar_prev, "y_traj": y_traj, "t_traj": t_traj}

    def walk_jump(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
    ):
        out = self.walk(score_fn, batch_size=batch_size, y_init=y_init)
        y, y_traj, t_traj = out["y_bar"], out["y_traj"], out["t_traj"]

        xhat = xhat_fn(score_fn, y, sigma=self.sigma / math.sqrt(self.m), beta=self.beta)

        if y_traj is not None:
            sigma_traj = self.sigma / t_traj.pow(0.5)
            xhat_traj = torch.stack(
                [
                    xhat_fn(
                        score_fn, y_traj[i].to(y.device), sigma=sigma_traj[i].item(), beta=self.beta
                    ).to(y_traj.device)
                    for i in tqdm(
                        range(y_traj.size(0)),
                        leave=False,
                        desc="jump",
                        disable=not self.verbose,
                    )
                ],
                dim=0,
            )
        else:
            xhat_traj = None

        return {
            "xhat": xhat,
            "xhat_traj": xhat_traj,
            "t_traj": t_traj,
        }

    def sample(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
        g_score_fn: Callable | None = None,
    ):
        out = self.walk_jump(score_fn, batch_size=batch_size, y_init=y_init)
        out["sample"] = out["xhat"]
        for key in out.keys():
            if isinstance(out[key], Tensor):
                out[key] = out[key].to("cpu")
        return out
