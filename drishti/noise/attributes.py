"""Lexicon-based attribute-phrase parser for person descriptions.

A description is split into units (words, with multi-word colours and garments merged into one unit).
An attribute is a garment / accessory noun plus the colour and pattern words directly before it, e.g.
"light blue striped shirt" -> noun "shirt", colours ["light blue"], patterns ["striped"].

The parser is deterministic and needs no model downloads, so noisy test sets are exactly reproducible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

RES = Path(__file__).parent / "resources"
_TOKEN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+|[^\sA-Za-z\d]")


@dataclass
class Attribute:
    noun_idx: int
    noun: str
    category: str
    salient: bool
    color_idxs: list[int] = field(default_factory=list)   # colour words and light/dark modifiers
    pattern_idxs: list[int] = field(default_factory=list)
    joiner_idxs: list[int] = field(default_factory=list)  # "and" inside "black and white"

    @property
    def mod_idxs(self) -> list[int]:
        return sorted(self.color_idxs + self.pattern_idxs + self.joiner_idxs)


@lru_cache(maxsize=None)
def load_lexicon(path: str | None = None) -> dict:
    lex = yaml.safe_load(Path(path or RES / "lexicon.yaml").read_text(encoding="utf-8"))
    lex["colors"] = set(lex["colors"]) | set(lex["multiword_colors"])
    lex["color_modifiers"] = set(lex["color_modifiers"])
    lex["patterns"] = set(lex["patterns"])
    lex["_multi"] = sorted({w for w in list(lex["multiword_colors"]) + list(lex["garments"]) if " " in w},
                           key=lambda w: -len(w.split()))
    return lex


def tokenize(text: str, lex: dict) -> list[str]:
    """Split text into units, merging multi-word colours / garments (e.g. 'navy blue', 't shirt')."""
    toks = _TOKEN.findall(text)
    units, i = [], 0
    while i < len(toks):
        for phrase in lex["_multi"]:
            n = len(phrase.split())
            if " ".join(t.lower() for t in toks[i:i + n]) == phrase:
                units.append(" ".join(toks[i:i + n]))
                i += n
                break
        else:
            units.append(toks[i])
            i += 1
    return units


def detokenize(units: list[str]) -> str:
    text = " ".join(u for u in units if u)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;])(\s*[,;.])+", lambda m: m.group(0)[-1], text)       # ", ," -> ","
    text = re.sub(r"\b(and|with|a|an|the|of)\s*(?=[,.;]|$)", "", text)        # dangling words
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"^[\s,;.]+", "", text)

    def fix_article(m: re.Match) -> str:
        art, word = m.group(1), m.group(2)
        lw = word.lower()
        vowel = lw[0] in "aeiou" and not lw.startswith(("uni", "use", "one", "eu"))
        new = "an" if vowel else "a"
        return (new.capitalize() if art[0].isupper() else new) + " " + word

    text = re.sub(r"\b(a|an|A|An) ([A-Za-z]+)", fix_article, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse(text: str, lex: dict | None = None) -> tuple[list[str], list[Attribute]]:
    lex = lex or load_lexicon()
    units = tokenize(text, lex)
    low = [u.lower() for u in units]
    attrs: list[Attribute] = []
    for i, w in enumerate(low):
        info = lex["garments"].get(w)
        if info is None:
            continue
        a = Attribute(noun_idx=i, noun=w, category=info["category"], salient=bool(info["salient"]))
        j = i - 1
        while j >= 0:
            if low[j] in lex["colors"] or low[j] in lex["color_modifiers"]:
                a.color_idxs.append(j)
            elif low[j] in lex["patterns"]:
                a.pattern_idxs.append(j)
            elif low[j] in ("and", "-", "/") and j - 1 >= 0 and low[j - 1] in lex["colors"]:
                a.joiner_idxs.append(j)
            else:
                break
            j -= 1
        a.color_idxs.sort()
        a.pattern_idxs.sort()
        attrs.append(a)
    return units, attrs
