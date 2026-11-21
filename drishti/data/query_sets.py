"""Query sets: JSON files of text queries used for evaluation.

Format:
    {"name": "commission_s2", "language": "en", "queries": [{"text": "...", "pid": 12, "k": 2}, ...]}

The clean test queries come straight from the dataset; noisy and Indian-language versions are produced
by scripts/make_noisy_sets.py and scripts/make_indic_queries.py with the same order and pids.
"""
from __future__ import annotations

import json
from pathlib import Path


def save_query_set(path: str | Path, name: str, queries: list[dict], language: str = "en", **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "language": language, **meta, "queries": queries},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def load_query_set(path: str | Path) -> dict:
    qs = json.loads(Path(path).read_text(encoding="utf-8"))
    if "queries" not in qs:
        raise ValueError(f"{path} is not a query set (missing 'queries')")
    qs.setdefault("name", Path(path).stem)
    return qs
