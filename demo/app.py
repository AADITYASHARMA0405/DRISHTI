"""Gradio demo: type a witness description, get the top-10 matching people.

    python demo/app.py --config configs/mvp_robust.yaml \
        --checkpoint outputs/mvp_robust_seed1/best.pt --gallery outputs/mvp_robust_seed1/gallery.npz \
        [--compare-checkpoint outputs/mvp_baseline_seed1/best.pt --compare-gallery outputs/mvp_baseline_seed1/gallery.npz]

Build the gallery file first with `eval.py --save-gallery`. Non-English queries are translated to English
with IndicTrans2 (translate-then-search), which needs requirements-indic.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from drishti.config import load_config  # noqa: E402
from drishti.index import GallerySearcher  # noqa: E402
from drishti.lang import LANGUAGES, QueryNormaliser  # noqa: E402
from drishti.models import load_model  # noqa: E402
from drishti.utils import get_device  # noqa: E402

EXAMPLES = [
    ["A woman wearing a red saree with a golden border, carrying a black handbag.", "en"],
    ["The man wears a white shirt, blue jeans and black shoes and carries a backpack.", "en"],
    ["ಕೆಂಪು ಸೀರೆ ಉಟ್ಟ ಮಹಿಳೆ, ಕೈಯಲ್ಲಿ ಕಪ್ಪು ಚೀಲ", "kn"],
    ["लाल कुर्ता और नीली जींस पहने एक आदमी", "hi"],
]


class Engine:
    def __init__(self, cfg, checkpoint, gallery, device):
        self.model = load_model(checkpoint, cfg, device)
        self.searcher = GallerySearcher(gallery)
        self.device = device

    @torch.no_grad()
    def search(self, text_en: str, k: int):
        q = self.model.encode_text([text_en]).cpu().numpy()
        scores, idx = self.searcher.search(q, k)
        return [(self.searcher.image_path(int(i)), f"#{r + 1}  score {s:.3f}  pid {int(self.searcher.pids[i])}")
                for r, (s, i) in enumerate(zip(scores[0], idx[0]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--compare-checkpoint", default=None)
    ap.add_argument("--compare-gallery", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    import gradio as gr

    cfg = load_config(a.config)
    device = get_device()
    main_engine = Engine(cfg, a.checkpoint, a.gallery, device)
    cmp_engine = (Engine(cfg, a.compare_checkpoint, a.compare_gallery or a.gallery, device)
                  if a.compare_checkpoint else None)
    norm = QueryNormaliser()

    def run(text: str, lang: str):
        if not text.strip():
            return "Type a description first.", [], []
        t0 = time.perf_counter()
        en = norm.to_english(text, lang)
        t1 = time.perf_counter()
        res = main_engine.search(en, a.k)
        t2 = time.perf_counter()
        info = (f"**Query used for search (English):** {en}  \n"
                f"translation {1000 * (t1 - t0):.0f} ms · search {1000 * (t2 - t1):.0f} ms · "
                f"{len(main_engine.searcher)} gallery images · index: {main_engine.searcher.backend}")
        return info, res, (cmp_engine.search(en, a.k) if cmp_engine else [])

    with gr.Blocks(title="DRISHTI person search") as ui:
        gr.Markdown("## DRISHTI: witness-robust person search\nDescribe the person as the witness remembers. "
                    "Results are a shortlist for human review, not an identification.")
        with gr.Row():
            text = gr.Textbox(label="Witness description (press Enter to search)", lines=1, max_lines=4,
                              scale=4)
            lang = gr.Dropdown(choices=[(v, k) for k, v in LANGUAGES.items()], value="en", label="Language",
                               scale=1)
        btn = gr.Button("Search", variant="primary")
        info = gr.Markdown()
        main_gal = gr.Gallery(label=f"Model: {Path(a.checkpoint).parent.name}", columns=10, height=330)
        cmp_gal = gr.Gallery(label=f"Compare: {Path(a.compare_checkpoint).parent.name}" if cmp_engine
                             else "Compare (not loaded)", columns=10, height=330, visible=cmp_engine is not None)
        gr.Examples(EXAMPLES, inputs=[text, lang])
        btn.click(run, [text, lang], [info, main_gal, cmp_gal])
        text.submit(run, [text, lang], [info, main_gal, cmp_gal])
    ui.launch(share=a.share)


if __name__ == "__main__":
    main()
