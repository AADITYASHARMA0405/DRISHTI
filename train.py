"""Train a DRISHTI model.

    python train.py --config configs/mvp_baseline.yaml
    python train.py --config configs/mvp_robust.yaml --seed 1 --opts train.batch_size=32 train.grad_accum=2
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drishti.config import load_config, output_dir
from drishti.data import PedesTrain, collate_train, gallery_and_queries, load_records
from drishti.evaluation import encode_gallery, evaluate_queries
from drishti.models import DualEncoder, save_checkpoint, total_loss
from drishti.utils import RunLogger, get_device, set_seed


def noise_ratio_for_epoch(epoch: int, cfg: dict) -> float:
    """Curriculum: clean for `clean_epochs`, then linear from start_ratio to end_ratio."""
    n = cfg.get("noise", {}) or {}
    if not n.get("enabled"):
        return 0.0
    E, c = cfg["train"]["epochs"], int(n.get("clean_epochs", 0))
    if epoch < c:
        return 0.0
    span = max(E - c - 1, 1)
    t = min((epoch - c) / span, 1.0)
    return float(n.get("start_ratio", 0.1)) + t * (float(n.get("end_ratio", 0.5)) - float(n.get("start_ratio", 0.1)))


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--opts", nargs="*", default=[], help="overrides like train.epochs=5")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.opts)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["run_name"] = f"{cfg['run_name']}_seed{args.seed}"
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    set_seed(cfg["seed"])
    device = get_device()
    print(f"[train] run={cfg['run_name']} device={device} out={out}")

    model = DualEncoder(cfg).to(device)
    ds = PedesTrain(cfg, seed=cfg["seed"])
    tc = cfg["train"]
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["data"].get("num_workers", 0), collate_fn=collate_train,
                    pin_memory=device.type == "cuda", persistent_workers=cfg["data"].get("num_workers", 0) > 0)
    opt = torch.optim.AdamW(model.param_groups(float(tc["lr"]), float(tc["lr_new"]),
                                               float(tc["weight_decay"])))
    accum = int(tc.get("grad_accum", 1))
    steps_per_epoch = max(len(dl) // accum, 1)
    sched = cosine_with_warmup(opt, int(tc.get("warmup_epochs", 0) * steps_per_epoch),
                               tc["epochs"] * steps_per_epoch)
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_gallery = val_queries = None
    if cfg["eval"].get("every_epoch"):
        val_gallery, val_queries = gallery_and_queries(load_records(cfg["data"]["json"], cfg["eval"]["split"]))
        val_gpids = torch.tensor([g["pid"] for g in val_gallery])

    logger = RunLogger(out, cfg)
    best, step, history = -1.0, 0, []
    for epoch in range(tc["epochs"]):
        ratio = noise_ratio_for_epoch(epoch, cfg)
        ds.set_epoch(epoch, ratio)
        model.train()
        t0, running = time.time(), {}
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(tqdm(dl, desc=f"epoch {epoch + 1}/{tc['epochs']} noise={ratio:.2f}")):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                img = model.encode_image(batch["image"])
                txt = model.encode_text(batch["text"])
            loss, parts = total_loss(model, batch, img.float(), txt.float(), cfg)
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            for k, v in {"loss": float(loss.detach()), **parts}.items():
                running[k] = running.get(k, 0.0) + v
        rec = {"epoch": epoch + 1, "noise_ratio": ratio, "time_s": round(time.time() - t0, 1),
               **{f"train_{k}": v / max(len(dl), 1) for k, v in running.items()}}
        if val_gallery is not None:
            model.eval()
            g = encode_gallery(model, val_gallery, cfg, device, progress=False)
            m = evaluate_queries(model, val_queries, g, val_gpids, device, progress=False)
            rec.update({f"val_{k}": v for k, v in m.items() if k != "n_queries"})
            if m["R1"] > best:
                best = m["R1"]
                save_checkpoint(model, out / "best.pt", epoch=epoch + 1, val=m)
        save_checkpoint(model, out / "last.pt", epoch=epoch + 1)
        logger.log(rec, step=step)
        history.append(rec)
        print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in rec.items()))
    logger.close()
    return {"out_dir": str(out), "history": history, "best_val_R1": best}


if __name__ == "__main__":
    main()
