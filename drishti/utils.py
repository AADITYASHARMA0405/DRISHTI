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


def get_rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KeepAwake:
    """Ask Windows not to sleep while training runs (the call video players use).

    On "Modern Standby" laptops the system drops into standby as soon as the screen turns off for
    inactivity, so the display is kept on as well. No system setting is changed; the request ends when the
    block exits or the process ends. Closing the lid or a low battery still sleeps the machine. Does nothing
    on other operating systems.
    """

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002

    def __init__(self, keep_display_on: bool = True):
        self.flags = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        if keep_display_on:
            self.flags |= self.ES_DISPLAY_REQUIRED

    def __enter__(self):
        import sys
        self._k32 = None
        if sys.platform == "win32":
            import ctypes
            self._k32 = ctypes.windll.kernel32
            self._k32.SetThreadExecutionState(self.flags)
        return self

    def __exit__(self, *exc):
        if self._k32 is not None:
            self._k32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def on_battery() -> bool:
    """True when a Windows laptop is running on battery (False on mains power, desktops or other OSes)."""
    import sys
    if sys.platform != "win32":
        return False
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    s = _Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return False
    return s.ACLineStatus == 0


def wait_for_mains(check=on_battery, poll_s: float = 30.0, log=print) -> float:
    """Block while the laptop is on battery so training never drains it into a forced sleep.
    Returns the seconds spent waiting."""
    if not check():
        return 0.0
    t0 = time.time()
    log("[power] on battery: training paused until the charger is connected")
    while check():
        time.sleep(poll_s)
    waited = time.time() - t0
    log(f"[power] mains power back after {waited / 60:.1f} min: resuming")
    return waited


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
