"""get_translator() must fall back to NLLB-200 when IndicTrans2 cannot be loaded (e.g. gated licence)."""
from drishti.lang import translate as T


class _Dummy:
    name = "dummy NLLB"

    def __init__(self, device=None):
        pass

    def translate(self, texts, src, tgt, **kw):
        return [f"{tgt}:{t}" for t in texts]


def test_falls_back_when_indictrans2_is_gated(monkeypatch):
    def gated(device=None):
        raise OSError("You are trying to access a gated repo.")

    monkeypatch.setattr(T, "IndicTrans2Translator", gated)
    monkeypatch.setattr(T, "NLLBTranslator", _Dummy)
    logs = []
    tr = T.get_translator(log=logs.append)
    assert tr.name == "dummy NLLB"
    assert any("falling back" in m for m in logs)
    assert tr.translate(["red saree"], "en", "kn") == ["kn:red saree"]


def test_prefers_indictrans2_when_available(monkeypatch):
    class _IT2(_Dummy):
        name = "IndicTrans2"

    monkeypatch.setattr(T, "IndicTrans2Translator", _IT2)
    monkeypatch.setattr(T, "NLLBTranslator", _Dummy)
    assert T.get_translator(log=lambda m: None).name == "IndicTrans2"
