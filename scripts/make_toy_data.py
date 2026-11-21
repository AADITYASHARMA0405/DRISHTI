"""Create a small synthetic PEDES-style dataset for smoke tests (no real people).

Each "person" is an image whose top half is the shirt colour and bottom half the trouser colour, with a
caption like "a person wearing a red shirt and blue jeans carrying a black bag".

    python scripts/make_toy_data.py --out datasets/toy --persons 120
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 150, 50), "yellow": (230, 210, 40),
          "black": (20, 20, 20), "white": (235, 235, 235), "pink": (240, 130, 180), "orange": (240, 140, 30),
          "purple": (120, 50, 160), "grey": (128, 128, 128), "brown": (110, 70, 30), "maroon": (110, 20, 40)}
UPPER = ["shirt", "kurta", "t-shirt", "jacket"]
LOWER = ["jeans", "trousers", "salwar", "skirt"]
BAGS = ["bag", "backpack"]


def make(out: Path, persons: int, imgs_per_person: int = 2, seed: int = 0, size=(32, 64)) -> Path:
    rng = random.Random(seed)
    names = list(COLORS)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    records = []
    for pid in range(persons):
        up, lo, bag = rng.sample(names, 3)
        upg, log_, bagn = rng.choice(UPPER), rng.choice(LOWER), rng.choice(BAGS)
        split = "train" if pid < int(persons * 0.7) else ("val" if pid < int(persons * 0.85) else "test")
        for j in range(imgs_per_person):
            w, h = size
            img = Image.new("RGB", (w, h), COLORS[up])
            d = ImageDraw.Draw(img)
            d.rectangle([0, h // 2, w, h], fill=COLORS[lo])
            bx = rng.randint(0, w - 8)
            d.rectangle([bx, h // 3, bx + 7, h // 3 + 10], fill=COLORS[bag])
            px = img.load()
            for _ in range(60):  # pixel noise so images of one person differ
                x, y = rng.randrange(w), rng.randrange(h)
                px[x, y] = tuple(rng.randrange(256) for _ in range(3))
            rel = f"p{pid:04d}_{j}.png"
            img.save(out / "imgs" / rel)
            caps = [f"a person wearing a {up} {upg} and {lo} {log_} carrying a {bag} {bagn}",
                    f"this person has a {bag} {bagn}, a {up} {upg} and {lo} {log_}"]
            records.append({"split": split, "id": pid, "file_path": rel, "captions": caps})
    path = out / "toy_pedes.json"
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/toy")
    ap.add_argument("--persons", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("wrote", make(Path(a.out), a.persons, seed=a.seed))
