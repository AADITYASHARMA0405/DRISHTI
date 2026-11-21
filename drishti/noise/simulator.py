"""Witness-noise simulator: injects omission, vagueness and commission errors into descriptions.

Example
-------
>>> sim = WitnessNoiseSimulator()
>>> r = sim.corrupt("a woman in a red saree carrying a black bag", op="commission", severity=1,
...                 rng=random.Random(0))
>>> r.text, r.k, r.A        # e.g. ("a woman in a maroon saree carrying a black bag", 1, 2)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .attributes import RES, Attribute, detokenize, load_lexicon, parse

OPS = ("omission", "vagueness", "commission")
BASIC_COLORS = ["black", "white", "grey", "red", "blue", "green", "yellow", "orange", "pink",
                "purple", "brown", "maroon"]


@dataclass
class NoiseResult:
    text: str
    k: int                      # number of attributes corrupted
    A: int                      # number of attributes found in the original description
    edits: list[dict] = field(default_factory=list)


class WitnessNoiseSimulator:
    def __init__(self, lexicon_path: str | None = None, confusions_path: str | None = None,
                 peripheral_weight: float = 2.0):
        self.lex = load_lexicon(lexicon_path)
        conf = yaml.safe_load(Path(confusions_path or RES / "confusions.yaml").read_text(encoding="utf-8"))
        self.color_conf: dict = conf["color_confusions"]
        self.vague_colors: dict = conf["vague_colors"]
        self.vague_garments: dict = conf["vague_garments"]
        self.garment_group: dict[str, list[str]] = {}
        for group in conf["garment_groups"]:
            for g in group:
                self.garment_group[g] = [x for x in group if x != g]
        self.peripheral_weight = peripheral_weight

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _color_phrase(units, a: Attribute) -> str:
        return " ".join(units[i].lower() for i in a.color_idxs + a.joiner_idxs if units[i])

    @staticmethod
    def _replace_span(units, idxs: list[int], new: str) -> None:
        idxs = sorted(idxs)
        units[idxs[0]] = new
        for i in idxs[1:]:
            units[i] = ""

    # ------------------------------------------------------------------ operators
    def _omission(self, units, a: Attribute, rng) -> dict | None:
        if a.mod_idxs:
            before = " ".join(units[i] for i in a.mod_idxs + [a.noun_idx])
            for i in a.mod_idxs:
                units[i] = ""
            return {"op": "omission", "before": before, "after": units[a.noun_idx]}
        before = units[a.noun_idx]
        units[a.noun_idx] = ""
        if a.noun_idx > 0 and units[a.noun_idx - 1].lower() in ("a", "an", "the"):
            units[a.noun_idx - 1] = ""
        return {"op": "omission", "before": before, "after": ""}

    def _vagueness(self, units, a: Attribute, rng) -> dict | None:
        if a.color_idxs:
            phrase = self._color_phrase(units, a)
            words = phrase.split()
            coarse = self.vague_colors.get(phrase) or self.vague_colors.get(words[-1])
            if coarse is None and words[0] in ("light", "pale"):
                coarse = "light"
            if coarse is None and words[0] in ("dark", "deep"):
                coarse = "dark"
            if coarse and coarse != phrase:
                self._replace_span(units, a.color_idxs + a.joiner_idxs, coarse)
                return {"op": "vagueness", "before": phrase, "after": coarse}
        coarse = self.vague_garments.get(a.noun)
        if coarse:
            before = units[a.noun_idx]
            units[a.noun_idx] = coarse
            return {"op": "vagueness", "before": before, "after": coarse}
        return None

    def _commission(self, units, a: Attribute, rng) -> dict | None:
        if a.color_idxs:
            phrase = self._color_phrase(units, a)
            words = phrase.split()
            options = self.color_conf.get(phrase) or self.color_conf.get(words[-1])
            if not options:
                options = [c for c in BASIC_COLORS if c not in words]
            new = rng.choice(options)
            self._replace_span(units, a.color_idxs + a.joiner_idxs, new)
            return {"op": "commission", "before": phrase, "after": new}
        options = self.garment_group.get(a.noun)
        if options:
            new = rng.choice(options)
            before = units[a.noun_idx]
            units[a.noun_idx] = new
            return {"op": "commission", "before": before, "after": new}
        return None

    def _apply(self, op, units, a, rng):
        return {"omission": self._omission, "vagueness": self._vagueness,
                "commission": self._commission}[op](units, a, rng)

    # ------------------------------------------------------------------ public API
    def parse(self, text: str):
        return parse(text, self.lex)

    def corrupt(self, text: str, op: str = "mixed", severity: int = 1,
                rng: random.Random | None = None) -> NoiseResult:
        """Corrupt up to `severity` attributes. op: omission | vagueness | commission | mixed."""
        if op != "mixed" and op not in OPS:
            raise ValueError(f"unknown op {op!r}")
        rng = rng or random.Random()
        units, attrs = parse(text, self.lex)
        remaining = list(attrs)
        edits: list[dict] = []
        while remaining and len(edits) < severity:
            weights = [1.0 if a.salient else self.peripheral_weight for a in remaining]
            a = rng.choices(remaining, weights=weights, k=1)[0]
            remaining.remove(a)
            ops = [op] if op != "mixed" else rng.sample(OPS, k=len(OPS))
            for o in ops:
                e = self._apply(o, units, a, rng)
                if e:
                    e["noun"] = a.noun
                    edits.append(e)
                    break
        return NoiseResult(detokenize(units), k=len(edits), A=len(attrs), edits=edits)

    def mask_attributes(self, text: str, prob: float, rng: random.Random | None = None) -> NoiseResult:
        """Attribute masking for training: drop each attribute's words independently with `prob`."""
        rng = rng or random.Random()
        units, attrs = parse(text, self.lex)
        edits = [e for a in attrs if rng.random() < prob for e in [self._omission(units, a, rng)] if e]
        return NoiseResult(detokenize(units), k=len(edits), A=len(attrs), edits=edits)
