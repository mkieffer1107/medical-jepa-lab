from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class Tracker:
    """Rank-zero Trackio logging with an unconditional local JSONL record."""

    def __init__(
        self,
        backend: str,
        project: str,
        run_name: str,
        config: dict[str, Any],
        run_dir: Path,
        enabled: bool,
        space_id: str | None = None,
    ) -> None:
        self.enabled = enabled and backend != "none"
        self.backend = backend if self.enabled else "none"
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._trackio = None
        if self.backend == "trackio":
            LOGGER.info("Initializing Trackio: project=%s run=%s space=%s", project, run_name, space_id or "local")
            try:
                import trackio

                kwargs: dict[str, Any] = {
                    "project": project,
                    "name": run_name,
                    "config": config,
                }
                if space_id:
                    kwargs["space_id"] = space_id
                trackio.init(**kwargs)
                self._trackio = trackio
                LOGGER.info("Trackio ready")
            except Exception as exc:  # network/auth failure should not abort training
                LOGGER.warning("Trackio initialization failed; continuing with JSONL: %s", exc)
                self.backend = "jsonl"
        elif self.backend not in {"jsonl", "none"}:
            raise ValueError(f"Unknown tracking backend: {backend}")
        if self.enabled:
            LOGGER.info("Metrics backend=%s | local metrics=%s", self.backend, self.jsonl_path.resolve())

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        scalar_metrics = {k: _scalar(v) for k, v in metrics.items()}
        record = {"step": int(step), **scalar_metrics}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self._trackio is not None:
            try:
                self._trackio.log(scalar_metrics, step=step)
            except Exception as exc:
                LOGGER.warning("Trackio log failed at step %d: %s", step, exc)
                self._trackio = None

    def finish(self) -> None:
        if self._trackio is not None:
            try:
                self._trackio.finish()
            except Exception as exc:
                LOGGER.warning("Trackio finish failed: %s", exc)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value
