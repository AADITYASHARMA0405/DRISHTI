"""Feature extraction and evaluation helpers shared by train.py, eval.py and the demo."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import GalleryImages
from .metrics import retrieval_metrics


@torch.no_grad()
def encode_gallery(model, gallery: list[dict], cfg: dict, device, batch_size: int | None = None,
                   progress: bool = True) -> torch.Tensor:
    d = cfg["data"]
    ds = GalleryImages(gallery, d["image_root"], d["image_size"])
    dl = DataLoader(ds, batch_size=batch_size or cfg["eval"]["batch_size"], shuffle=False,
                    num_workers=d.get("num_workers", 0), pin_memory=device.type == "cuda")
    feats = []
    use_amp = device.type == "cuda"
    for x, _ in tqdm(dl, desc="gallery", disable=not progress):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats.append(model.encode_image(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(feats)


@torch.no_grad()
def encode_texts(model, texts: list[str], device, batch_size: int = 256, progress: bool = True) -> torch.Tensor:
    feats = []
    use_amp = device.type == "cuda"
    for s in tqdm(range(0, len(texts), batch_size), desc="queries", disable=not progress):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats.append(model.encode_text(texts[s:s + batch_size]).float().cpu())
    return torch.cat(feats)


def evaluate_queries(model, queries: list[dict], g_feats: torch.Tensor, g_pids: torch.Tensor, device,
                     progress: bool = True) -> dict:
    q = encode_texts(model, [x["text"] for x in queries], device, progress=progress)
    q_pids = torch.tensor([x["pid"] for x in queries])
    dev = device if device.type == "cuda" else torch.device("cpu")
    sim = q.to(dev) @ g_feats.to(dev).t()
    return retrieval_metrics(sim, q_pids, g_pids)
