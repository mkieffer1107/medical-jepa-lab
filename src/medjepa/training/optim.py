from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WarmupCosineScheduler:
    optimizer: torch.optim.Optimizer
    total_steps: int
    warmup_steps: int
    base_lr: float
    min_lr: float

    def value(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.base_lr
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    def step(self, step: int) -> float:
        lr = self.value(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_optimizer(module: nn.Module, cfg: object) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: object,
    steps_per_epoch: int,
    total_steps_override: int | None = None,
) -> WarmupCosineScheduler:
    total_steps = total_steps_override or int(cfg.epochs) * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs) * steps_per_epoch
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=float(cfg.learning_rate),
        min_lr=float(cfg.min_learning_rate),
    )


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + (end - start) * progress


def gradient_norm(module: nn.Module) -> float:
    squared = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return float(squared.sqrt().item())
