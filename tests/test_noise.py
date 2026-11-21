import random

import pytest

from drishti.noise import WitnessNoiseSimulator, parse
from drishti.noise.attributes import detokenize

TEXT = ("A woman wearing a red saree with a golden border, carrying a black bag "
        "and wearing white sandals.")


@pytest.fixture(scope="module")
def sim():
    return WitnessNoiseSimulator()


def test_parse_finds_attributes():
    units, attrs = parse(TEXT)
    found = {a.noun: [units[i].lower() for i in a.color_idxs] for a in attrs}
    assert found == {"saree": ["red"], "bag": ["black"], "sandals": ["white"]}
    assert {a.noun: a.salient for a in attrs} == {"saree": True, "bag": False, "sandals": False}


def test_multiword_colours_and_garments():
    units, attrs = parse("he wears a navy blue t shirt and light blue jeans")
    found = {a.noun: [units[i].lower() for i in a.color_idxs] for a in attrs}
    assert found == {"t shirt": ["navy blue"], "jeans": ["light blue"]}


@pytest.mark.parametrize("op", ["omission", "vagueness", "commission", "mixed"])
@pytest.mark.parametrize("sev", [1, 2, 3])
def test_severity_counts(sim, op, sev):
    r = sim.corrupt(TEXT, op=op, severity=sev, rng=random.Random(7))
    assert r.A == 3
    assert r.k == sev
    assert r.text != TEXT


def test_deterministic_with_seed(sim):
    a = sim.corrupt(TEXT, "mixed", 2, random.Random("x-1")).text
    b = sim.corrupt(TEXT, "mixed", 2, random.Random("x-1")).text
    assert a == b


def test_omission_removes_colour(sim):
    r = sim.corrupt("a man in a red shirt", "omission", 1, random.Random(0))
    assert r.text == "a man in a shirt"


def test_commission_uses_confusion_table(sim):
    r = sim.corrupt("a man in a red shirt", "commission", 1, random.Random(0))
    assert r.edits[0]["after"] in {"maroon", "orange", "pink"}


def test_vagueness_coarsens(sim):
    r = sim.corrupt("a man in a navy blue jacket", "vagueness", 1, random.Random(0))
    assert r.text == "a man in a dark jacket"


def test_no_attributes_is_safe(sim):
    r = sim.corrupt("a person walking on the road", "mixed", 3, random.Random(0))
    assert (r.k, r.A) == (0, 0)
    assert r.text == "a person walking on the road"


def test_article_fix():
    assert detokenize(["a", "orange", "kurta"]) == "an orange kurta"
    assert detokenize(["an", "red", "uniform"]) == "a red uniform"
