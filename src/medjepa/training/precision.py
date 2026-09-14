from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


class PrecisionManager:
    def __init__(self, requested: str, device: torch.device) -> None:
        requested = requested.lower()
        if requested not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unknown precision: {requested}")
        if device.type != "cuda" and requested == "fp16":
            requested = "fp32"
        if device.type == "cuda" and requested == "bf16" and not torch.cuda.is_bf16_supported():
            requested = "fp32"
        self.name = requested
        self.device = device
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(requested)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and requested == "fp16"
        )

    def autocast(self) -> ContextManager:
        if self.dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def prepare_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        module: torch.nn.Module,
        clip_norm: float | None,
    ) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(optimizer)
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), clip_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler.is_enabled():
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
