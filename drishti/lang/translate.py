"""IndicTrans2 translation (AI4Bharat) between English, Hindi and Kannada.

Uses the distilled 200M checkpoints from Hugging Face:
    ai4bharat/indictrans2-en-indic-dist-200M   (English -> Indic)
    ai4bharat/indictrans2-indic-en-dist-200M   (Indic -> English)
Requires `pip install -r requirements-indic.txt`. Language codes follow FLORES-200.
"""
from __future__ import annotations

import torch

LANG_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
MODELS = {"en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
          "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M"}


ISO = {"eng_Latn": "en", "hin_Deva": "hi", "kan_Knda": "kn"}
DEVANAGARI = {"hi"}


class FallbackProcessor:
    """Pure-Python stand-in for IndicTransToolkit's IndicProcessor (which needs a C++ compiler on Windows).

    Mirrors the IndicTrans2 recipe: punctuation normalisation, Indic normalisation, script unification of
    non-Devanagari Indic text to Devanagari, word tokenisation, and the "src_lang tgt_lang" tag prefix;
    post-processing reverses the script unification and detokenises. Use the official toolkit (e.g. on
    Kaggle) for the query sets reported in the paper.
    """

    def __init__(self):
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_detokenize, indic_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

        self.punc = MosesPunctNormalizer()
        self.en_tok, self.en_detok = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
        self.norm = {l: IndicNormalizerFactory().get_normalizer(l) for l in ("hi", "kn")}
        self.tok, self.detok, self.xlit = indic_tokenize, indic_detokenize, UnicodeIndicTransliterator

    def preprocess_batch(self, batch, src_lang, tgt_lang, **_):
        iso, out = ISO[src_lang], []
        for s in batch:
            s = self.punc.normalize(s.strip())
            if iso == "en":
                s = " ".join(self.en_tok.tokenize(s, escape=False))
            else:
                s = " ".join(self.tok.trivial_tokenize(self.norm[iso].normalize(s), iso))
                if iso not in DEVANAGARI:
                    s = self.xlit.transliterate(s, iso, "hi")
            out.append(f"{src_lang} {tgt_lang} {s}")
        return out

    def postprocess_batch(self, sents, lang, **_):
        iso, out = ISO[lang], []
        for s in sents:
            if iso == "en":
                out.append(self.en_detok.detokenize(s.split()))
            else:
                if iso not in DEVANAGARI:
                    s = self.xlit.transliterate(s, "hi", iso)
                out.append(self.detok.trivial_detokenize(s, iso))
        return out


def _indic_processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor
        return IndicProcessor(inference=True), "IndicTransToolkit"
    except ImportError:
        pass
    try:
        from IndicTransToolkit import IndicProcessor
        return IndicProcessor(inference=True), "IndicTransToolkit"
    except ImportError:
        pass
    try:
        return FallbackProcessor(), "fallback (indic-nlp-library)"
    except ImportError as e:
        raise ImportError("Install the Indic tools: pip install -r requirements-indic.txt") from e


class IndicTranslator:
    """translator.translate(["a man in a red shirt"], "en", "kn") -> ["ಕೆಂಪು ಶರ್ಟ್ ..."]"""

    def __init__(self, direction: str, device: str | None = None, fp16: bool = True):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if direction not in MODELS:
            raise ValueError(f"direction must be one of {list(MODELS)}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = MODELS[direction]
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(self.device).eval()
        if fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.ip, self.processor_name = _indic_processor()

    @torch.no_grad()
    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int = 32,
                  num_beams: int = 5, max_length: int = 256) -> list[str]:
        s, t = LANG_CODES[src], LANG_CODES[tgt]
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = self.ip.preprocess_batch(texts[i:i + batch_size], src_lang=s, tgt_lang=t)
            enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                           return_attention_mask=True).to(self.device)
            gen = self.model.generate(**enc, use_cache=True, min_length=0, max_length=max_length,
                                      num_beams=num_beams, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang=t))
        return out
