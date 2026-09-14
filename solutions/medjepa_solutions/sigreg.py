from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn import functional as dist_nn


def invariance_loss(projections: torch.Tensor) -> torch.Tensor:
    if projections.ndim != 3:
        raise ValueError("Expected projections=(views,batch,dimension)")
    center = projections.mean(dim=0, keepdim=True)
    return (projections - center).square().mean()


def _differentiable_global_mean(local_mean: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return local_mean
    summed = dist_nn.all_reduce(local_mean, op=dist.ReduceOp.SUM)
    return summed / dist.get_world_size()


class SIGReg(nn.Module):
    """Sliced Epps-Pulley-style regularization toward an isotropic N(0, I)."""

    def __init__(
        self,
        num_slices: int,
        t_max: float,
        num_points: int,
        seed: int,
    ) -> None:
        super().__init__()
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        self.num_slices = int(num_slices)
        self.seed = int(seed)
        t = torch.linspace(0.0, float(t_max), int(num_points), dtype=torch.float32)
        dt = float(t_max) / (int(num_points) - 1)
        integration_weights = torch.full_like(t, 2.0 * dt)
        integration_weights[[0, -1]] = dt
        normal_cf = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("normal_cf", normal_cf)
        self.register_buffer("weights", integration_weights * normal_cf)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _projection_matrix(self, dimension: int, device: torch.device) -> torch.Tensor:
        step = self.global_step.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(step, op=dist.ReduceOp.MAX)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + int(step.item()))
        matrix = torch.randn(
            dimension,
            self.num_slices,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.global_step.copy_(step + 1)
        return matrix

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        if projections.ndim < 2:
            raise ValueError("Expected (..., samples, dimensions)")
        samples = projections.float()
        local_count = samples.shape[-2]
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        global_count = local_count * world_size
        directions = self._projection_matrix(samples.shape[-1], samples.device)
        sliced = samples @ directions
        phases = sliced.unsqueeze(-1) * self.t
        cosine_mean = torch.cos(phases).mean(dim=-3)
        sine_mean = torch.sin(phases).mean(dim=-3)
        cosine_mean = _differentiable_global_mean(cosine_mean)
        sine_mean = _differentiable_global_mean(sine_mean)
        error = (cosine_mean - self.normal_cf).square() + sine_mean.square()
        statistic = (error * self.weights).sum(dim=-1) * global_count
        return statistic.mean()


class LeJEPAObjective(nn.Module):
    def __init__(self, sigreg: SIGReg, regularization_weight: float) -> None:
        super().__init__()
        if not 0 <= regularization_weight <= 1:
            raise ValueError("regularization_weight must lie in [0, 1]")
        self.sigreg = sigreg
        self.regularization_weight = float(regularization_weight)

    def forward(self, projections: torch.Tensor) -> dict[str, torch.Tensor]:
        invariance = invariance_loss(projections)
        regularization = self.sigreg(projections)
        weight = self.regularization_weight
        total = (1.0 - weight) * invariance + weight * regularization
        return {
            "loss": total,
            "invariance": invariance,
            "sigreg": regularization,
        }
