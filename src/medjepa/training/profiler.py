from __future__ import annotations

from pathlib import Path

import torch


class NullProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self) -> None:
        return None


def make_profiler(cfg: object, run_dir: Path, rank: int):
    if not bool(getattr(cfg, "enabled", False)):
        return NullProfiler()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_dir = run_dir / "profiler" / f"rank-{rank}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(getattr(cfg, "wait", 2)),
            warmup=int(getattr(cfg, "warmup", 2)),
            active=int(getattr(cfg, "active", 4)),
            repeat=int(getattr(cfg, "repeat", 1)),
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
