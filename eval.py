"""Evaluate a checkpoint on the clean test split and any number of query sets; append to results.csv.

    python eval.py --config configs/mvp_robust.yaml --checkpoint outputs/mvp_robust_seed1/best.pt \
        --query-sets query_sets/cuhk_pedes/*.json --save-gallery outputs/mvp_robust_seed1/gallery.npz

Without --checkpoint the zero-shot pre-trained model is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from drishti.config import load_config
from drishti.data import gallery_and_queries, load_query_set, load_records
from drishti.evaluation import encode_gallery, evaluate_queries
from drishti.models import load_model
from drishti.utils import get_device

FIELDS = ["time", "run", "checkpoint", "seed", "query_set", "language", "R1", "R5", "R10", "mAP", "mINP",
          "n_queries"]


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--query-sets", nargs="*", default=[], help="query-set JSON files (globs allowed)")
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--run", default=None, help="name for this model in results.csv")
    ap.add_argument("--save-gallery", default=None, help="save gallery features to .npz for the demo")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.opts)
    device = get_device()
    model = load_model(args.checkpoint, cfg, device)
    run = args.run or (Path(args.checkpoint).parent.name if args.checkpoint else f"{cfg['run_name']}_zeroshot")

    gallery, clean = gallery_and_queries(load_records(cfg["data"]["json"], args.split))
    g_pids = torch.tensor([g["pid"] for g in gallery])
    g_feats = encode_gallery(model, gallery, cfg, device)
    if args.save_gallery:
        Path(args.save_gallery).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_gallery, feats=g_feats.numpy().astype("float32"), pids=g_pids.numpy(),
                 paths=np.array([g["path"] for g in gallery]), image_root=str(cfg["data"]["image_root"]))
        print(f"[eval] saved gallery features to {args.save_gallery}")

    sets = [{"name": "clean", "language": "en", "queries": clean}]
    for pattern in args.query_sets:
        for f in sorted(glob.glob(pattern)) or [pattern]:
            sets.append(load_query_set(f))

    rows = []
    for qs in sets:
        m = evaluate_queries(model, qs["queries"], g_feats, g_pids, device)
        row = {"time": datetime.now().isoformat(timespec="seconds"), "run": run,
               "checkpoint": args.checkpoint or "zero-shot", "seed": cfg.get("seed"),
               "query_set": qs["name"], "language": qs.get("language", "en"),
               **{k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()}}
        rows.append(row)
        print(f"[eval] {qs['name']:<22} R1={m['R1']:.2f} R5={m['R5']:.2f} R10={m['R10']:.2f} "
              f"mAP={m['mAP']:.2f} mINP={m['mINP']:.2f}")

    res = Path(args.results)
    res.parent.mkdir(parents=True, exist_ok=True)
    new = not res.exists()
    with res.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[eval] appended {len(rows)} rows to {res}")
    return rows


if __name__ == "__main__":
    main()
