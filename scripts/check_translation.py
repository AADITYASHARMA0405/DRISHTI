"""Sanity-check IndicTrans2 on person descriptions: English -> Hindi / Kannada -> English, plus romanization.

    python scripts/check_translation.py --out outputs/translation_check.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.lang.romanize import Romanizer  # noqa: E402
from drishti.lang.translate import get_translator  # noqa: E402

SENTENCES = [
    "A woman wearing a red saree with a golden border, carrying a black handbag.",
    "The man is wearing a white kurta, blue jeans and black sandals.",
    "A young girl in a pink salwar kameez with a yellow dupatta and a school bag.",
    "He has short black hair and wears a grey jacket over a striped shirt.",
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/translation_check.txt")
    ap.add_argument("--translator", default="indictrans2", choices=["indictrans2", "nllb"])
    a = ap.parse_args(argv)
    lines = []
    t0 = time.time()
    fwd = back = get_translator(prefer=a.translator)
    rom = Romanizer()
    lines.append(f"translator: {fwd.name}; romanizer: {rom.backend}; models loaded in {time.time() - t0:.0f}s\n")
    for L in ("hi", "kn"):
        t = time.time()
        native = fwd.translate(SENTENCES, "en", L)
        english = back.translate(native, L, "en")
        lines.append(f"=== {L} ({time.time() - t:.1f}s for {len(SENTENCES)} sentences, both directions) ===")
        for src, nat, en in zip(SENTENCES, native, english):
            r = rom.to_roman(nat, L)
            lines += [f"EN    : {src}", f"{L.upper()}    : {nat}", f"ROMAN : {r}",
                      f"BACK  : {back.translate([rom.from_roman(r, L)], L, 'en')[0]}  (via romanized)",
                      f"BACK  : {en}  (via native)", ""]
    text = "\n".join(lines)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
