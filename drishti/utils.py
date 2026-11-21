"""Seeding, device selection and a small logger that writes JSONL and, optionally, Weights & Biases."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunLogger:
    """Logs metrics to <out_dir>/log.jsonl and to W&B when enabled and installed."""

    def __init__(self, out_dir: Path, cfg: dict):
        self.path = Path(out_dir) / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        wcfg = cfg.get("wandb", {}) or {}
        if wcfg.get("enabled"):
            try:
                import wandb

                self.wandb = wandb.init(project=wcfg.get("project", "drishti"),
                                        name=cfg.get("run_name"), config=cfg)
            except Exception as e:  # never let logging crash training
                print(f"[logger] W&B disabled: {e}")

    def log(self, metrics: dict, step: int | None = None) -> None:
        rec = {"time": round(time.time(), 1), "step": step, **metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
