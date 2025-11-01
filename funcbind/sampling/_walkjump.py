from typing import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm


class SingleMeasurementSampler:
    def __init__(
        self,
        mcmc,
        sigma: float = 1.0,
        beta: float = 1.0,
        y_init_distribution: torch.distributions.Distribution | None = None,
        **kwargs,
    ):
        self.mcmc = mcmc
        self.sigma = float(sigma)
        self.y_init_distribution = y_init_distribution
        self.beta = beta

    def walk(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
    ):
        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError(
                    "either y_init and y_init_distribution must be supplied"
                )
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,))

        y, _, y_traj, _ = self.mcmc(y_init, lambda y: score_fn(y, self.sigma))

        if y_traj is not None:
            t_traj = torch.ones(y_traj.size(0), device=y_traj.device, dtype=int)
        else:
            t_traj = None

        return {"y": y,
                "y_traj": y_traj,
                "t_traj": t_traj}

    def walk_jump(
        self,
        score_fn: Callable,
        batch_size: int | None = None,
        y_init: Tensor | None = None,
    ):
        out = self.walk(
            score_fn,
            batch_size=batch_size,
            y_init=y_init,
        )
        y, y_traj, t_traj = out["y"], out["y_traj"], out["t_traj"]

        xhat = y + (self.sigma**2) * score_fn(y, sigma=self.sigma)

        if y_traj is not None:
            xhat_traj = torch.stack(
                [
                    y_traj[i, :].to(y.device)
                    + (self.sigma**2) * score_fn(y_traj[i, :], sigma=self.sigma)
                    for i in tqdm(range(y_traj.size(0)), leave=False, desc="jump")
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
