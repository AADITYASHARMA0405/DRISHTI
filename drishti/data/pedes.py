"""Loaders for PEDES-style datasets (CUHK-PEDES, ICFG-PEDES, RSTPReid and the toy set).

Annotation format (a JSON list), as distributed with CUHK-PEDES `reid_raw.json`:
    {"split": "train" | "val" | "test", "id": <person id>, "file_path": "<relative image path>",
     "captions": ["...", "..."]}
RSTPReid uses "img_path" instead of "file_path"; both are accepted.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..noise import OPS, WitnessNoiseSimulator

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transform(image_size, train: bool) -> T.Compose:
    size = tuple(image_size)
    ops = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        ops += [T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if train:
        ops += [T.RandomErasing(p=0.3, scale=(0.02, 0.2), value=0)]
    return T.Compose(ops)


def load_records(json_path: str | Path, split: str) -> list[dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = []
    for r in data:
        if r.get("split") != split:
            continue
        out.append({"pid": int(r["id"]), "path": r.get("file_path") or r["img_path"],
                    "captions": [c for c in r["captions"] if c and c.strip()]})
    if not out:
        raise ValueError(f"No records with split={split!r} in {json_path}")
    return out


def gallery_and_queries(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Gallery = unique images; queries = every caption, each with the pid of its image."""
    gallery, seen = [], set()
    for r in records:
        if r["path"] not in seen:
            seen.add(r["path"])
            gallery.append({"path": r["path"], "pid": r["pid"]})
    queries = [{"text": c, "pid": r["pid"]} for r in records for c in r["captions"]]
    return gallery, queries


class PedesTrain(Dataset):
    """One item per (image, caption). Applies the witness-noise curriculum and attribute masking."""

    def __init__(self, cfg: dict, seed: int = 0):
        d = cfg["data"]
        self.root = Path(d["image_root"])
        recs = load_records(d["json"], "train")
        self.items = [(r["path"], r["pid"], c) for r in recs for c in r["captions"]]
        self.tf = build_transform(d["image_size"], train=True)
        self.sim = WitnessNoiseSimulator()
        ncfg = cfg.get("noise", {}) or {}
        self.ops = list(ncfg.get("ops", OPS))
        self.max_sev = int(ncfg.get("max_severity", 3))
        self.mask_prob = float((cfg.get("masking", {}) or {}).get("prob", 0.0))
        self.noise_ratio = 0.0
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int, noise_ratio: float) -> None:
        self.epoch, self.noise_ratio = epoch, noise_ratio

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        path, pid, cap = self.items[i]
        img = self.tf(Image.open(self.root / path).convert("RGB"))
        rng = random.Random(f"{self.seed}-{self.epoch}-{i}")
        k, A = 0, None
        if self.noise_ratio > 0 and rng.random() < self.noise_ratio:
            r = self.sim.corrupt(cap, op=rng.choice(self.ops), severity=rng.randint(1, self.max_sev), rng=rng)
            cap, k, A = r.text, r.k, r.A
        elif self.mask_prob > 0:
            r = self.sim.mask_attributes(cap, self.mask_prob, rng=rng)
            cap, k, A = r.text, r.k, r.A
        if A is None:
            A = len(self.sim.parse(cap)[1])
        return {"image": img, "text": cap, "pid": pid, "k": k, "A": A}


def collate_train(batch: list[dict]) -> dict:
    return {"image": torch.stack([b["image"] for b in batch]),
            "text": [b["text"] for b in batch],
            "pid": torch.tensor([b["pid"] for b in batch]),
            "k": torch.tensor([b["k"] for b in batch], dtype=torch.float),
            "A": torch.tensor([b["A"] for b in batch], dtype=torch.float)}


class GalleryImages(Dataset):
    def __init__(self, gallery: list[dict], image_root: str | Path, image_size):
        self.g, self.root = gallery, Path(image_root)
        self.tf = build_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.g)

    def __getitem__(self, i: int):
        return self.tf(Image.open(self.root / self.g[i]["path"]).convert("RGB")), i
