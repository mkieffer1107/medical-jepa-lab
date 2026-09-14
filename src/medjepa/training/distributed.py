from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedEnvironment:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(timeout_minutes: int = 30) -> DistributedEnvironment:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return DistributedEnvironment(distributed, rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def wrap_ddp(module: nn.Module, env: DistributedEnvironment) -> nn.Module:
    if not env.distributed:
        return module
    kwargs: dict[str, object] = {
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    if env.device.type == "cuda":
        kwargs.update(device_ids=[env.local_rank], output_device=env.local_rank)
    return DistributedDataParallel(module, **kwargs)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def reduce_mean(value: torch.Tensor, env: DistributedEnvironment) -> torch.Tensor:
    result = value.detach().clone()
    if env.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= env.world_size
    return result


def reduce_sum(value: torch.Tensor, env: DistributedEnvironment) -> torch.Tensor:
    result = value.detach().clone()
    if env.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def reduce_max(value: torch.Tensor, env: DistributedEnvironment) -> torch.Tensor:
    """Return the slowest rank's detached scalar/tensor value."""
    result = value.detach().clone()
    if env.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result
