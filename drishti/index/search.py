"""Nearest-neighbour search over gallery embeddings. Uses FAISS when installed, otherwise NumPy."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class GallerySearcher:
    """Load gallery features saved by `eval.py --save-gallery` and search them with query embeddings."""

    def __init__(self, npz_path: str | Path, use_faiss: bool = True):
        z = np.load(npz_path, allow_pickle=False)
        self.feats = np.ascontiguousarray(z["feats"].astype("float32"))
        self.pids = z["pids"]
        self.paths = [str(p) for p in z["paths"]]
        self.image_root = str(z["image_root"]) if "image_root" in z.files else ""
        self.index = None
        if use_faiss:
            try:
                import faiss

                self.index = faiss.IndexFlatIP(self.feats.shape[1])   # inner product = cosine (normalised)
                self.index.add(self.feats)
            except ImportError:
                self.index = None
        self.backend = "faiss" if self.index is not None else "numpy"

    def __len__(self) -> int:
        return len(self.paths)

    def search(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """q: (Q, D) L2-normalised query features. Returns (scores, indices), each (Q, k)."""
        q = np.ascontiguousarray(np.atleast_2d(q).astype("float32"))
        k = min(k, len(self))
        if self.index is not None:
            return self.index.search(q, k)
        sims = q @ self.feats.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx

    def image_path(self, i: int) -> str:
        return str(Path(self.image_root) / self.paths[i]) if self.image_root else self.paths[i]
