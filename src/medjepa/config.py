from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A tiny recursively attribute-accessible dictionary."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Config):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(self)


def _as_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _as_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_config(v) for v in value]
    return value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    cfg = _as_config(raw)
    for item in overrides:
        apply_override(cfg, item)
    validate_config(cfg)
    return cfg


def apply_override(cfg: Config, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must look like section.key=value, got: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    value = yaml.safe_load(raw_value)
    keys = dotted_key.split(".")
    cursor: Config = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], Config):
            cursor[key] = Config()
        cursor = cursor[key]
    cursor[keys[-1]] = _as_config(value)


def validate_config(cfg: Config) -> None:
    required = ["experiment", "algorithm", "data", "model", "optimization", "tracking"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if cfg.algorithm not in {"ijepa", "lejepa"}:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
    image_size = int(cfg.data.image_size)
    patch_size = int(cfg.model.patch_size)
    if image_size % patch_size:
        raise ValueError("data.image_size must be divisible by model.patch_size")
    if int(cfg.model.embed_dim) % int(cfg.model.num_heads):
        raise ValueError("model.embed_dim must be divisible by model.num_heads")


def clone_config(cfg: Config) -> Config:
    return _as_config(copy.deepcopy(cfg.to_dict()))
