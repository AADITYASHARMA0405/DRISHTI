"""YAML config loading with `base:` inheritance and command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config. `base: other.yaml` is resolved relative to the file.

    overrides: strings like "train.epochs=5" or "data.image_size=[224,224]"; values are parsed as YAML.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got: {ov!r}")
        key, val = ov.split("=", 1)
        _set_nested(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def output_dir(cfg: dict) -> Path:
    return Path(cfg.get("output_root", "outputs")) / cfg.get("run_name", "run")
