from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)


@dataclass
class StepTimer:
    previous: float = field(default_factory=time.perf_counter)

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = now - self.previous
        self.previous = now
        return elapsed


def feature_diagnostics(features: torch.Tensor) -> dict[str, float]:
    """Cheap collapse diagnostics for a `(samples, dimensions)` tensor."""
    x = features.detach().float()
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    centered = x - x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    norms = x.norm(dim=-1)
    if x.shape[0] > 1:
        covariance = centered.T @ centered / x.shape[0]
        diagonal = torch.diag(torch.diag(covariance))
        off_diagonal_rms = (covariance - diagonal).square().mean().sqrt()
    else:
        off_diagonal_rms = torch.zeros((), device=x.device)
    return {
        "features/std_mean": float(std.mean().item()),
        "features/std_min": float(std.min().item()),
        "features/norm_mean": float(norms.mean().item()),
        "features/offdiag_cov_rms": float(off_diagonal_rms.item()),
    }
