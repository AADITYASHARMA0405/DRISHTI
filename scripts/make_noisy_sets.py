"""Build the clean and Noisy-Witness query sets for a split (MVP acceptance criterion AC2).

    python scripts/make_noisy_sets.py --config configs/mvp_baseline.yaml --out query_sets/cuhk_pedes

Writes clean.json and <op>_s<severity>.json for op in {omission, vagueness, commission, mixed} and
severity in {1, 2, 3}. Every file is fully determined by --seed: re-running gives identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.config import load_config                                      # noqa: E402
from drishti.data import gallery_and_queries, load_records, save_query_set  # noqa: E402
from drishti.noise import WitnessNoiseSimulator                             # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ops", nargs="*", default=["omission", "vagueness", "commission", "mixed"])
    ap.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.opts)
    _, queries = gallery_and_queries(load_records(cfg["data"]["json"], a.split))
    out = Path(a.out)
    sim = WitnessNoiseSimulator()
    save_query_set(out / "clean.json", "clean", queries, split=a.split)
    summary = {}
    for op in a.ops:
        for sev in a.severities:
            noisy, ks = [], []
            for i, q in enumerate(queries):
                r = sim.corrupt(q["text"], op=op, severity=sev, rng=random.Random(f"{a.seed}-{op}-{sev}-{i}"))
                noisy.append({"text": r.text, "pid": q["pid"], "k": r.k, "A": r.A, "clean": q["text"]})
                ks.append(r.k)
            name = f"{op}_s{sev}"
            save_query_set(out / f"{name}.json", name, noisy, split=a.split, op=op, severity=sev, seed=a.seed)
            changed = sum(k > 0 for k in ks) / len(ks)
            summary[name] = {"mean_k": sum(ks) / len(ks), "changed": changed}
            print(f"{name:<16} queries={len(noisy)} changed={100 * changed:.1f}% mean_k={summary[name]['mean_k']:.2f}")
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(out.glob("*.json")))).hexdigest()[:16]
    print(f"checksum of all query sets: {digest}  (identical across runs with the same seed)")
    return {"summary": summary, "checksum": digest}


if __name__ == "__main__":
    main()
