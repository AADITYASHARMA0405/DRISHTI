"""Check that CLIP runs on this GPU and find the largest batch size that fits for training.

    python scripts/check_gpu.py --config configs/mvp_robust.yaml
    python scripts/check_gpu.py --config configs/local/rtx4050_robust.yaml --batch-sizes 16 32 48 64

For each batch size it runs one full training step (forward, SDM + robust loss, backward, optimiser step)
on random images and real-looking captions, and reports peak GPU memory and step time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from drishti.config import load_config  # noqa: E402
from drishti.models import DualEncoder, total_loss  # noqa: E402

CAPTION = ("A woman wearing a red saree with a golden border, carrying a black handbag and wearing "
           "white sandals. She has long black hair tied in a plait.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[16, 32, 48, 64])
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: install the CUDA build of PyTorch (see README).")

    cfg = load_config(a.config, a.opts)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  memory: {props.total_memory / 2**30:.1f} GiB")
    t0 = time.time()
    model = DualEncoder(cfg).to(dev).train()
    print(f"model loaded in {time.time() - t0:.1f}s  "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"image size: {cfg['data']['image_size']}  grad checkpointing: {cfg['model'].get('grad_checkpointing', False)}")
    opt = torch.optim.AdamW(model.param_groups(1e-5, 1e-3, 0.02))
    scaler = torch.amp.GradScaler("cuda")
    H, W = cfg["data"]["image_size"]

    for bs in a.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = {"image": torch.randn(bs, 3, H, W, device=dev), "text": [CAPTION] * bs,
                     "pid": torch.arange(bs, device=dev), "k": torch.zeros(bs, device=dev),
                     "A": torch.full((bs,), 3.0, device=dev)}
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t = time.time()
                with torch.autocast("cuda"):
                    img = model.encode_image(batch["image"])
                    txt = model.encode_text(batch["text"])
                loss, _ = total_loss(model, batch, img.float(), txt.float(), cfg)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"batch {bs:>3}: OK   peak {peak:.2f} GiB   step {min(times[1:]):.2f}s")
        except torch.OutOfMemoryError:
            print(f"batch {bs:>3}: OUT OF MEMORY")
            opt.zero_grad(set_to_none=True)
            break


if __name__ == "__main__":
    main()
