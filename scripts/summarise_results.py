"""Turn outputs/results.csv into a paper-style table: mean ± std over seeds, plus robustness retention.

    python scripts/summarise_results.py --results outputs/results.csv --out outputs/summary.md

Runs named <name>_seed<k> are grouped as <name>. Retention = R1(query set) / R1(clean) for that model.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def fmt(vals: list[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{st.mean(vals):.2f} ± {st.stdev(vals):.2f}"


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.csv")
    ap.add_argument("--out", default="outputs/summary.md")
    ap.add_argument("--metric", default="R1")
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(Path(a.results).open(encoding="utf-8")))
    data: dict = defaultdict(lambda: defaultdict(list))     # model -> query_set -> [values]
    for r in rows:
        model = re.sub(r"_seed\d+$", "", r["run"])
        data[model][r["query_set"]].append(float(r[a.metric]))
    sets = sorted({qs for m in data.values() for qs in m}, key=lambda s: (s != "clean", s))
    lines = [f"| Model | " + " | ".join(sets) + " |", "|" + "---|" * (len(sets) + 1)]
    for model, per in data.items():
        lines.append(f"| {model} | " + " | ".join(fmt(per[s]) if s in per else "–" for s in sets) + " |")
    lines += ["", f"Robustness retention ({a.metric} on set / {a.metric} on clean, mean over seeds):", "",
              "| Model | " + " | ".join(s for s in sets if s != "clean") + " |",
              "|" + "---|" * len(sets)]
    for model, per in data.items():
        if "clean" not in per:
            continue
        c = st.mean(per["clean"])
        lines.append(f"| {model} | " + " | ".join(
            f"{st.mean(per[s]) / c:.3f}" if s in per else "–" for s in sets if s != "clean") + " |")
    text = "\n".join(lines) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    main()
