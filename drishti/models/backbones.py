"""Image-text backbones. Each exposes encode_image(pixels) and encode_text(list[str]) -> (B, D) tensors.

hf_clip : Hugging Face CLIP (default openai/clip-vit-base-patch16). Used for real experiments.
toy     : a tiny CNN + bag-of-words encoder for CPU smoke tests; no downloads.
"""
from __future__ import annotations

import re
import zlib

import torch
import torch.nn as nn


def _features(out, dim: int) -> torch.Tensor:
    """transformers versions differ in what get_*_features returns; normalise to a (B, dim) tensor."""
    if torch.is_tensor(out):
        return out
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        t = getattr(out, name, None)
        if torch.is_tensor(t) and t.shape[-1] == dim:
            return t
    raise RuntimeError("Could not read projected features from the CLIP output; "
                       "check your transformers version.")


class HFClipBackbone(nn.Module):
    def __init__(self, name: str, image_size, max_text_len: int = 77, grad_checkpointing: bool = False):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.clip = CLIPModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.max_len = max_text_len
        self.embed_dim = self.clip.config.projection_dim
        native = self.clip.config.vision_config.image_size
        self.interpolate = tuple(image_size) != (native, native)
        if grad_checkpointing:
            self.clip.gradient_checkpointing_enable()

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        kw = {"interpolate_pos_encoding": True} if self.interpolate else {}
        return _features(self.clip.get_image_features(pixel_values=pixels, **kw), self.embed_dim)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        dev = next(self.parameters()).device
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(dev)
        return _features(self.clip.get_text_features(**tok), self.embed_dim)


class ToyBackbone(nn.Module):
    """Small, fast and download-free. Good enough to learn colour/garment words on synthetic data."""

    def __init__(self, embed_dim: int = 128, vocab: int = 4096, max_text_len: int = 32):
        super().__init__()
        self.embed_dim, self.vocab, self.max_len = embed_dim, vocab, max_text_len
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(), nn.Linear(64 * 8, embed_dim))
        self.emb = nn.EmbeddingBag(vocab, embed_dim, mode="mean", padding_idx=0)
        self.txt = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim))

    def _ids(self, text: str) -> list[int]:
        words = re.findall(r"[a-z]+", text.lower())[: self.max_len]
        return [1 + zlib.crc32(w.encode()) % (self.vocab - 1) for w in words] or [0]

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.cnn(pixels)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        dev = self.emb.weight.device
        ids = [self._ids(t) for t in texts]
        flat = torch.tensor([i for x in ids for i in x], device=dev)
        offsets = torch.tensor([0] + [len(x) for x in ids[:-1]], device=dev).cumsum(0)
        return self.txt(self.emb(flat, offsets))


def build_backbone(cfg: dict) -> nn.Module:
    m, d = cfg["model"], cfg["data"]
    if m["backbone"] == "hf_clip":
        return HFClipBackbone(m["name"], d["image_size"], m.get("max_text_len", 77),
                              m.get("grad_checkpointing", False))
    if m["backbone"] == "toy":
        return ToyBackbone(max_text_len=m.get("max_text_len", 32))
    raise ValueError(f"unknown backbone {m['backbone']!r}")
