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
from drishti.utils import (KeepAwake, RunLogger, get_device, get_rng_state, set_rng_state, set_seed,
                           wait_for_mains)


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
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out_dir>/resume.pt if it exists (safe to pass on a fresh run)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop cleanly after this many epochs in this run (continue later with --resume)")
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
    # Workers persist across epochs (starting them costs minutes on Windows); the curriculum state they read
    # lives in shared memory, see PedesTrain.set_epoch. The shuffle order is re-seeded every epoch so a
    # resumed run sees the same batches as an uninterrupted one.
    nw = cfg["data"].get("num_workers", 0)
    shuffle_gen = torch.Generator()
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True, num_workers=nw,
                    collate_fn=collate_train, pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                    generator=shuffle_gen)
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

    best, step, history, start_epoch = -1.0, 0, [], 0
    resume_path = out / "resume.pt"
    if args.resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        best, step, history, start_epoch = st["best"], st["step"], st["history"], st["epoch"]
        set_rng_state(st["rng"])
        print(f"[train] resumed after epoch {start_epoch} (best val R1 {best:.2f})")

    logger = RunLogger(out, cfg)
    pause_on_battery = bool(tc.get("pause_on_battery", True))
    with KeepAwake(keep_display_on=bool(tc.get("keep_display_on", True))):
        for epoch in range(start_epoch, tc["epochs"]):
            ratio = noise_ratio_for_epoch(epoch, cfg)
            ds.set_epoch(epoch, ratio)
            shuffle_gen.manual_seed(cfg["seed"] * 1000 + epoch)
            model.train()
            t0, running = time.time(), {}
            opt.zero_grad(set_to_none=True)
            for it, batch in enumerate(tqdm(dl, desc=f"epoch {epoch + 1}/{tc['epochs']} noise={ratio:.2f}")):
                if pause_on_battery and it % 25 == 0:
                    t0 += wait_for_mains(log=lambda m: print(m, flush=True))   # paused time is not epoch time
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
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
            history.append(rec)
            # write to a temporary file first so a crash mid-save never leaves a broken resume.pt
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                        "step": step, "best": best, "history": history, "rng": get_rng_state()},
                       out / "resume.tmp")
            (out / "resume.tmp").replace(resume_path)
            logger.log(rec, step=step)
            print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in rec.items()), flush=True)
            if args.stop_after and epoch + 1 - start_epoch >= args.stop_after and epoch + 1 < tc["epochs"]:
                print(f"[train] stopping after epoch {epoch + 1} as requested; continue with --resume")
                logger.close()
                return {"out_dir": str(out), "history": history, "best_val_R1": best, "stopped": True}
    logger.close()
    resume_path.unlink(missing_ok=True)          # 1.8 GB of optimizer state is no longer needed
    (out / ".done").write_text(json.dumps({"best_val_R1": best, "epochs": tc["epochs"]}), encoding="utf-8")
    return {"out_dir": str(out), "history": history, "best_val_R1": best}


if __name__ == "__main__":
    main()
