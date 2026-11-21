# DRISHTI: witness-robust multilingual person search

RV College of Engineering · ANN and Deep Learning (AI253IA)
Team: Aaditya Sharma · Dia Arora · Kathan Gajera

DRISHTI searches a gallery of people (CCTV crops) using a witness's description, in English, Hindi,
Kannada or romanized text, and stays accurate when the description is incomplete or partly wrong. This
repository holds the **MVP** described in `7_MVP_Specification`: CLIP fine-tuned on CUHK-PEDES, a
witness-noise simulator, a robust training recipe, translate-then-search for Indian languages, and a Gradio
demo.

## 1. Set-up

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux / Kaggle: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU build
pip install -r requirements.txt
pip install -r requirements-indic.txt   # only for Hindi / Kannada queries
python -m pytest -q                     # 26 tests, ~1 min on CPU, no downloads
```

### Hugging Face access for IndicTrans2 (each member, once)
The IndicTrans2 models are **gated**. Each member must:
1. Create a free account at huggingface.co and open both model pages:
   `ai4bharat/indictrans2-en-indic-dist-200M` and `ai4bharat/indictrans2-indic-en-dist-200M`.
2. Click **"Agree and access repository"** on each page.
3. Create a *read* access token (Settings → Access Tokens) and log in on your machine:
   `.venv\Scripts\hf auth login` (paste the token yourself; never commit it).

Then check translation with `python scripts/check_translation.py`.

### Windows notes
* `IndicTransToolkit` needs Microsoft C++ Build Tools to compile on Windows, so it is skipped there and
  `drishti/lang/translate.py` uses an equivalent pure-Python processor. Generate the query sets reported in
  the paper on Kaggle / Linux, where the official toolkit installs normally.
* If C: is nearly full, keep caches on another drive:
  `set HF_HOME=D:\ANN\.hf-cache` and `set PIP_CACHE_DIR=D:\ANN\.pip-cache`.

## 2. Data

Request **CUHK-PEDES** from its authors (Li et al., CVPR 2017) and place it as:

```
datasets/CUHK-PEDES/reid_raw.json
datasets/CUHK-PEDES/imgs/...
```

Then point the config at it, either by editing `configs/base.yaml` or on the command line:
`--opts data.json=datasets/CUHK-PEDES/reid_raw.json data.image_root=datasets/CUHK-PEDES/imgs`.
ICFG-PEDES and RSTPReid use the same loader (RSTPReid's `img_path` key is also accepted).

## 3. MVP workflow (each step maps to an acceptance criterion)

```bash
# AC2: clean + noisy query sets (deterministic; prints a checksum)
python scripts/make_noisy_sets.py --config configs/mvp_baseline.yaml --out query_sets/cuhk_pedes

# AC5: Hindi / Kannada / romanized versions and their translate-then-search English versions (GPU advised)
python scripts/make_indic_queries.py --in query_sets/cuhk_pedes/clean.json --out query_sets/cuhk_pedes

# Zero-shot CLIP reference
python eval.py --config configs/mvp_baseline.yaml --query-sets "query_sets/cuhk_pedes/*.json"

# AC1 / AC4: train baseline and robust model with 3 seeds each
python train.py --config configs/mvp_baseline.yaml --seed 1      # repeat with --seed 2, 3
python train.py --config configs/mvp_robust.yaml   --seed 1      # repeat with --seed 2, 3

# AC3 / AC4 / AC5: evaluate every checkpoint on every query set
python eval.py --config configs/mvp_robust.yaml --checkpoint outputs/mvp_robust_seed1/best.pt \
    --query-sets "query_sets/cuhk_pedes/*.json" --save-gallery outputs/mvp_robust_seed1/gallery.npz

# Paper-style table: mean ± std over seeds and robustness retention
python scripts/summarise_results.py

# AC6: demo (compare robust vs baseline side by side)
python demo/app.py --config configs/mvp_robust.yaml \
    --checkpoint outputs/mvp_robust_seed1/best.pt --gallery outputs/mvp_robust_seed1/gallery.npz \
    --compare-checkpoint outputs/mvp_baseline_seed1/best.pt --compare-gallery outputs/mvp_baseline_seed1/gallery.npz
```

**On a 6 GB laptop GPU (RTX 4050)** use `configs/local/rtx4050_*.yaml`, which halve the batch, accumulate
gradients and enable gradient checkpointing. On Kaggle / Colab (16 GB) use the normal configs.

**AC1 note:** our trainer is CLIP + SDM loss. If its Rank-1 is not within 2 points of the published
TBPS-CLIP number, run the official TBPS-CLIP code (github.com/Flame-Chasers/TBPS-CLIP) for the baseline
and report both; the evaluation harness here only needs a checkpoint that produces embeddings.

## 4. Repository map and owners

| Path | What it does | Owner |
|---|---|---|
| `drishti/data/` | PEDES loaders, noise curriculum in the training set, query-set files | Dia |
| `drishti/noise/` | Witness-noise simulator; extend `resources/*.yaml` to add words and confusions | Dia |
| `drishti/lang/` | IndicTrans2 translation, romanization, translate-then-search | Dia |
| `drishti/models/` | CLIP / toy backbones, dual encoder, SDM and robust triplet losses | Aaditya |
| `train.py` | Training loop: AMP, AdamW, cosine schedule, curriculum, checkpoints | Aaditya |
| `drishti/metrics.py`, `eval.py` | Rank-k, mAP, mINP; results.csv | Kathan |
| `drishti/index/`, `demo/` | FAISS search and Gradio demo | Kathan |
| `scripts/` | Data preparation and result summaries | Shared |
| `tests/` | Unit tests and an end-to-end smoke test on synthetic data | Shared |

## 5. Key ideas in the code

* **Witness errors** (`drishti/noise`): attributes are parsed with a lexicon (colours, Indian and Western
  garments, accessories), then corrupted by *omission* (forgotten), *vagueness* ("navy blue" → "dark") or
  *commission* (confident mistake: "red" → "maroon", "saree" → "lehenga"). Peripheral items (footwear,
  small bags) are corrupted more often than salient ones, matching eyewitness-memory findings.
* **Robust soft-margin triplet loss** (`drishti/models/losses.py`): the margin for a caption with *k* of
  *A* attributes corrupted is `m0 · (1 − k/A)`, so partly wrong descriptions are not over-penalised.
* **Curriculum** (`train.py`): clean epochs first, then the fraction of corrupted captions rises from
  10% to 50%.

## 6. Rules for results that go into the paper

1. Every number comes from `results.csv` produced by `eval.py`, never typed by hand.
2. Report mean ± std over at least 3 seeds.
3. Never tune on the test split; use `val` (the training loop already does).
4. Keep the noisy query sets fixed (same seed) for every model you compare.
