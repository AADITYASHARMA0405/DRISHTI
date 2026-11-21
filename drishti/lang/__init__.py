"""Indian-language tools. Imported lazily so the core package works without them."""

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "hi_rom": "Hindi (romanized)",
             "kn_rom": "Kannada (romanized)"}


class QueryNormaliser:
    """Turns a query in any supported form into English for the translate-then-search path."""

    def __init__(self):
        self._tr = None
        self._rom = None

    def to_english(self, text: str, lang: str) -> str:
        if lang == "en":
            return text
        if lang.endswith("_rom"):
            if self._rom is None:
                from .romanize import Romanizer
                self._rom = Romanizer()
            lang = lang[:-4]
            text = self._rom.from_roman(text, lang)
        if self._tr is None:
            from .translate import IndicTranslator
            self._tr = IndicTranslator("indic-en")
        return self._tr.translate([text], lang, "en")[0]
