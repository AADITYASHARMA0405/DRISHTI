"""Create Hindi, Kannada and romanized versions of a query set, plus their translate-then-search versions.

    python scripts/make_indic_queries.py --in query_sets/cuhk_pedes/clean.json --out query_sets/cuhk_pedes

For each language L in {hi, kn} this writes:
    L.json          the query in native script (for the multilingual model in Phase 3)
    L_rom.json      the same query romanized
    L_t2s.json      native script translated back to English (translate-then-search baseline, MVP AC5)
    L_rom_t2s.json  romanized -> native script -> English
These are machine translations: they simulate Indian-language queries for the MVP. Natively written
witness queries are collected in Phase 2. Each file records which translator produced it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.data import load_query_set, save_query_set  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=["hi", "kn"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (quick tests)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-romanized", action="store_true")
    ap.add_argument("--translator", default="indictrans2", choices=["indictrans2", "nllb"],
                    help="IndicTrans2 falls back to NLLB-200 automatically if its licence is not accepted")
    a = ap.parse_args(argv)

    from drishti.lang.translate import get_translator

    qs = load_query_set(a.inp)
    queries = qs["queries"][: a.limit] if a.limit else qs["queries"]
    texts = [q["text"] for q in queries]
    out = Path(a.out)
    base = qs["name"]

    tr = get_translator(prefer=a.translator)
    native = {L: tr.translate(texts, "en", L, batch_size=a.batch_size) for L in a.langs}

    rom = None
    if not a.no_romanized:
        from drishti.lang.romanize import Romanizer
        rom = Romanizer()
        print(f"romanizer backend: {rom.backend}")

    back = tr
    for L in a.langs:
        tag = f"{base}_{L}" if base != "clean" else L

        def save(name, txts, language, **meta):
            save_query_set(out / f"{name}.json", name,
                           [{**q, "text": t, "en": q["text"]} for q, t in zip(queries, txts)], language,
                           translator=tr.name, **meta)

        save(tag, native[L], L)
        save(f"{tag}_t2s", back.translate(native[L], L, "en", batch_size=a.batch_size), "en",
             pipeline=f"en->{L}->en")
        if rom:
            romanized = [rom.to_roman(t, L) for t in native[L]]
            save(f"{tag}_rom", romanized, f"{L}_rom", romanizer=rom.backend)
            renative = [rom.from_roman(t, L) for t in romanized]
            save(f"{tag}_rom_t2s", back.translate(renative, L, "en", batch_size=a.batch_size), "en",
                 pipeline=f"en->{L}->roman->{L}->en")
        print(f"[{L}] example: {texts[0]!r} -> {native[L][0]!r}")


if __name__ == "__main__":
    main()
