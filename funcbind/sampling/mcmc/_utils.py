import math
from typing import Callable

import torch
from torch import Tensor


def initialize_velocity(v_init: str | Tensor, y: Tensor, u: float) -> Tensor:
    """Initialize velocity according to the given method."""
    if isinstance(v_init, str):
        if v_init == "gaussian":
            return math.sqrt(u) * torch.randn_like(y)
        if v_init == "zero":
            return torch.zeros_like(y)
        raise RuntimeError(f"{v_init} not in (gaussian, zero)")

    if isinstance(v_init, torch.Tensor):
        return v_init

    raise RuntimeError(f"{type(v_init)=} must be either `str` or `Tensor`.")


def create_score_fn(
    score_fn: Callable, inverse_temperature: float, score_fn_clip: float | None = None
) -> Callable:
    """Create a score function that is clipped and scaled by the inverse temperature."""

    def score_fn_processed(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score function clipped and scaled by the inverse temperature."""
        orig_score = score_fn(y).to(dtype=y.dtype)
        # Clip the score by norm.
        score = orig_score
        if score_fn_clip is not None:
            norm = torch.linalg.vector_norm(score, dim=-1, keepdim=True)
            clip = torch.min(norm, torch.ones_like(norm) * score_fn_clip)
            score = (score / norm) * clip
        score = score * inverse_temperature
        return score, orig_score

    return score_fn_processed
