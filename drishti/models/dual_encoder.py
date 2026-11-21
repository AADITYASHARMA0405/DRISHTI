"""Dual encoder: backbone + temperature. Returns L2-normalised embeddings in the shared space."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone


class DualEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        m = cfg["model"]
        init = math.log(1.0 / float(m.get("temperature", 0.02)))
        self.logit_scale = nn.Parameter(torch.tensor(init), requires_grad=bool(m.get("learn_temperature")))

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone.encode_image(pixels).float(), dim=-1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self.backbone.encode_text(texts).float(), dim=-1)

    def param_groups(self, lr: float, lr_new: float, weight_decay: float) -> list[dict]:
        """Pre-trained backbone weights get `lr`; everything else (new layers, temperature) gets `lr_new`.
        Biases and norm weights get no weight decay."""
        pre_decay, pre_no, new_decay, new_no = [], [], [], []
        pretrained = self.cfg["model"]["backbone"] != "toy"
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_pre = pretrained and n.startswith("backbone.")
            no_decay = p.ndim < 2 or "norm" in n.lower() or n == "logit_scale"
            if is_pre:
                (pre_no if no_decay else pre_decay).append(p)
            else:
                (new_no if no_decay else new_decay).append(p)
        groups = [{"params": pre_decay, "lr": lr, "weight_decay": weight_decay},
                  {"params": pre_no, "lr": lr, "weight_decay": 0.0},
                  {"params": new_decay, "lr": lr_new, "weight_decay": weight_decay},
                  {"params": new_no, "lr": lr_new, "weight_decay": 0.0}]
        return [g for g in groups if g["params"]]


def save_checkpoint(model: DualEncoder, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, **extra}, path)


def load_model(checkpoint: str | Path | None, cfg: dict | None = None,
               device: torch.device | str = "cpu") -> DualEncoder:
    """Load from a checkpoint (its saved cfg is used unless cfg is given). With checkpoint=None,
    returns the zero-shot pre-trained model built from cfg."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
    model = DualEncoder(cfg or ckpt["cfg"])
    if ckpt:
        model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
