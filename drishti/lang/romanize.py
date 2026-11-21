"""Romanization (native script <-> Latin letters) for Hindi and Kannada.

Prefers AI4Bharat IndicXlit (natural spellings such as "kempu seere"). Falls back to the pure-Python
`indic-transliteration` package (ITRANS scheme, e.g. "kempu sIre"), which always installs and is exactly
reversible.
"""
from __future__ import annotations

XLIT_CODES = {"hi": "hi", "kn": "kn"}


class Romanizer:
    def __init__(self, prefer_xlit: bool = True):
        self.backend = None
        if prefer_xlit:
            try:
                from ai4bharat.transliteration import XlitEngine

                self._to_roman = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)
                self._from_roman = {l: XlitEngine(l, beam_width=4, rescore=False) for l in XLIT_CODES}
                self.backend = "indicxlit"
            except Exception:
                self.backend = None
        if self.backend is None:
            try:
                from indic_transliteration import sanscript  # noqa: F401
            except ImportError as e:
                raise ImportError("Install a romanizer: pip install indic-transliteration") from e
            self.backend = "itrans"

    def to_roman(self, text: str, lang: str) -> str:
        if self.backend == "indicxlit":
            return self._to_roman.translit_sentence(text, lang_code=XLIT_CODES[lang])
        from indic_transliteration import sanscript
        script = {"hi": sanscript.DEVANAGARI, "kn": sanscript.KANNADA}[lang]
        return sanscript.transliterate(text, script, sanscript.ITRANS)

    def from_roman(self, text: str, lang: str) -> str:
        if self.backend == "indicxlit":
            res = self._from_roman[lang].translit_sentence(text)
            return res[XLIT_CODES[lang]] if isinstance(res, dict) else res
        from indic_transliteration import sanscript
        script = {"hi": sanscript.DEVANAGARI, "kn": sanscript.KANNADA}[lang]
        return sanscript.transliterate(text, sanscript.ITRANS, script)
