# DRISHTI phase reports

One section per phase of the implementation plan (`6_Implementation_Plan_and_Work_Division`). Each section
says what was done, how, which tool serves which purpose, how it was verified, and what is still open.
Status words: **Done** (built and verified), **Ready** (built and tested on synthetic data; waits for the
real dataset), **Open** (not started or needs a person, not code).

---

## Phase 0 – Foundations (plan weeks 1–2)

### What was done

| Plan task | Owner | Status | How it was done |
|---|---|---|---|
| GitHub repository, folder structure, config system, W&B | Aaditya | **Done** | Git repo `drishti/`; YAML configs with `base:` inheritance and `--opts key=value` overrides (`drishti/config.py`); optional Weights & Biases logging plus a local `log.jsonl` (`drishti/utils.py`); `.gitignore` keeps datasets, checkpoints and outputs out of git |
| GPU environment | Aaditya | **Done** | Python 3.13 virtual environment on D: (`.venv`) with PyTorch 2.11 + CUDA 12.8; pip cache moved to D: because C: is 95% full; Kaggle notebook (`notebooks/kaggle_mvp.ipynb`) for 16 GB GPUs |
| PyTorch data loaders with standard splits | Dia | **Ready** | `drishti/data/pedes.py` reads the PEDES JSON format used by CUHK-PEDES (`reid_raw.json`), ICFG-PEDES and RSTPReid, and builds the standard gallery (unique images) and queries (every caption) |
| Evaluation script: Rank-1/5/10, mAP, mINP | Kathan | **Done** | `drishti/metrics.py` (vectorised, chunked so 6,000+ queries fit in memory) and `eval.py`, which appends every result to `outputs/results.csv` |
| Zero-shot CLIP on CUHK-PEDES test set | All | **Ready** | `python eval.py --config configs/mvp_baseline.yaml` (no checkpoint = zero-shot) |
| Ethics application, consent form, camera plan; IndoFashion access | Dia, Kathan | **Open** | Needs the team and the department, not code |
| Reading notes; faculty guide confirmed | All | **Open** | Team task |

### Exit criterion
*"All three members can train and evaluate zero-shot CLIP on CUHK-PEDES; ethics application submitted."*
Code side met. Still needs: CUHK-PEDES access (request it from the dataset authors) and the ethics
application.

---

## Phase 1 – MVP (plan weeks 2–6)

### Features (MVP specification, Table 2)

| ID | Feature | Owner | Status | Where | How it works |
|---|---|---|---|---|---|
| F1 | Data loaders | Dia | **Ready** | `drishti/data/pedes.py` | One training item per (image, caption); test split → gallery + queries with person ids |
| F2 | Noise simulator v1 | Dia | **Done** | `drishti/noise/` | Parses attribute phrases (colour/pattern words + garment noun) with a lexicon of 70+ Indian and Western garments; applies omission, vagueness or commission to 1–3 attributes; peripheral items (footwear, small bags) are chosen twice as often as salient ones; seeded, so every query set is byte-for-byte reproducible |
| F3 | Indic queries | Dia | **Ready** | `drishti/lang/`, `scripts/make_indic_queries.py` | IndicTrans2 English→Hindi/Kannada, romanization, then back to English for translate-then-search |
| F4 | Baseline model | Aaditya | **Ready** | `drishti/models/`, `configs/mvp_baseline.yaml` | CLIP ViT-B/16 dual encoder fine-tuned with the SDM loss (as in IRRA), 384×128 input via interpolated position embeddings |
| F5 | Robust model v1 | Aaditya | **Done** (code verified) | `drishti/models/losses.py`, `train.py`, `configs/mvp_robust.yaml` | Adds attribute masking (15%), the soft-margin triplet loss with margin `m0·(1 − k/A)`, and a curriculum: 2 clean epochs, then corrupted captions rise from 10% to 50% |
| F6 | Evaluation harness | Kathan | **Done** | `eval.py`, `scripts/summarise_results.py` | Evaluates clean + every query set in one run; summary gives mean ± std over seeds and robustness retention (noisy R1 ÷ clean R1) |
| F7 | Search demo | Kathan | **Ready** | `drishti/index/`, `demo/app.py` | Gallery embeddings saved by `eval.py --save-gallery`, searched with FAISS; Gradio page with language box and side-by-side robust vs. baseline results |

### Acceptance criteria (MVP specification, Table 3)

| ID | Criterion | Status | Evidence / how to check |
|---|---|---|---|
| AC1 | Reproduced baseline Rank-1 within 2 points of published | Ready | Needs CUHK-PEDES; run `train.py` with `mvp_baseline.yaml` |
| AC2 | Noisy test sets for 3 error types × 3 severities, identical from seeds | **Done** | `scripts/make_noisy_sets.py` prints a checksum; the test suite generates the sets twice and asserts identical checksums |
| AC3 | Baseline's clean → noisy Rank-1 drop measured | Ready | `eval.py` + `summarise_results.py` |
| AC4 | Robust model's drop smaller at every severity; clean R1 within 1 point | Ready | 3 seeds each; `summarise_results.py` |
| AC5 | Hindi / Kannada / romanized queries via translate-then-search | Ready | `make_indic_queries.py` writes `*_t2s.json` sets that `eval.py` scores |
| AC6 | Demo returns top-10 in under 1 second | Ready | The demo prints translation and search time per query |
| AC7 | A new member can reproduce from the README | Ready | README section 3 lists every command in order |

### Tools and what each is used for

| Tool | Purpose in DRISHTI | Used in |
|---|---|---|
| PyTorch 2.11 (CUDA 12.8) | Neural networks, training on the RTX 4050, mixed precision (AMP) | everywhere |
| Hugging Face Transformers | Pre-trained CLIP ViT-B/16 weights and tokenizer; loading IndicTrans2 | `models/backbones.py`, `lang/translate.py` |
| torchvision | Image resizing, flipping, random erasing, normalisation | `data/pedes.py` |
| IndicTrans2 + IndicTransToolkit (AI4Bharat) | English ↔ Hindi / Kannada translation | `lang/translate.py` |
| indic-transliteration / IndicXlit | Native script ↔ romanized text | `lang/romanize.py` |
| FAISS | Fast nearest-neighbour search over gallery embeddings | `index/search.py` |
| Gradio | Web demo for the faculty guide and the paper video | `demo/app.py` |
| Weights & Biases | Experiment tracking (losses, metrics, configs, commits) | `utils.py` |
| PyYAML | Configs and the editable lexicon / confusion tables | `config.py`, `noise/resources/` |
| pytest | Unit and end-to-end tests | `tests/` |
| NumPy | Saving gallery features for the demo | `eval.py`, `index/` |

### Deviations from the planning documents (and why)

* **Attribute parsing uses a lexicon, not spaCy.** A fixed lexicon makes noisy test sets exactly
  reproducible (AC2), needs no model download, and lets Dia add Indian garments by editing a YAML file.
  spaCy can still be added in Phase 3 for phrases such as "a shirt in red".
* **Code lives in a `drishti/` package** rather than flat folders, so imports work the same in scripts,
  tests and notebooks. The folders match the MVP specification one to one.

### Verification so far

* **Tests:** 26 / 26 pass, both on CPU and in the GPU environment (noise operators and determinism,
  metrics against a hand-computed example, both losses, and an end-to-end run).
* **End-to-end on synthetic data** (`scripts/make_toy_data.py`, toy network): on the RTX 4050, 16 epochs
  take about 25 s; test Rank-1 = 52.8, mAP = 62.6 (random ranking ≈ 8 and ≈ 13). The whole chain works:
  noise curriculum → training → checkpoint → evaluation → saved gallery → FAISS search.
* **Demo:** launched in the browser; an English query returned the top-10 in 559 ms (first query,
  including GPU warm-up) with the FAISS backend (AC6 met in form; re-time on the real gallery).
* **GPU check** (`scripts/check_gpu.py`, real CLIP ViT-B/16, 384×128 input, mixed precision, one full
  training step with both losses):

| Setting | Batch | Peak GPU memory | Step time | Verdict |
|---|---|---|---|---|
| No gradient checkpointing | 16 | 3.47 GiB | 1.84 s | fits |
| No gradient checkpointing | 32 | 4.95 GiB | 1.38 s | fits |
| No gradient checkpointing | 48 | 6.42 GiB | 5.35 s | exceeds 6 GB; Windows spills to system RAM |
| No gradient checkpointing | 64 | 7.78 GiB | 12.83 s | exceeds 6 GB; about 10× slower |
| **Gradient checkpointing** | **64** | **3.03 GiB** | **1.06 s** | **chosen for `configs/local/rtx4050_*.yaml`** |
| Gradient checkpointing | 96 | 3.57 GiB | 1.38 s | fits |

  So the laptop trains with the same batch size (64) as the paper runs. Estimated cost on CUHK-PEDES:
  ≈ 1,060 steps per epoch × ≈ 1.1 s ≈ 20 minutes per epoch (plus data loading), ≈ 3.5 hours per 10-epoch
  run, ≈ 21 hours for the six MVP runs (2 models × 3 seeds). Kaggle can run seeds in parallel.
* **Transformers 5.24 compatibility:** CLIP loads, position embeddings interpolate to 384×128, and the
  wrapper reads projected features correctly.
* **IndicTrans2:** the pre/post-processing (normalisation, Kannada ↔ Devanagari script unification, tags)
  is verified offline. The models themselves are **gated** on Hugging Face, so the download stopped with
  "401 – gated repo". Each member must accept the licence and log in once (README section 1); then
  `scripts/check_translation.py` runs the round-trip check.

### Environment decisions (and why)

| Decision | Why |
|---|---|
| Virtual environment and all caches (pip, Hugging Face) on D: | C: is 95% full; CUDA PyTorch alone is 2.7 GB |
| PyTorch 2.11.0 + CUDA 12.8, pinned | A plain `pip install` replaced it with a newer CPU-only build; pinning prevents that |
| IndicTransToolkit skipped on Windows, fallback processor added | It must be compiled with Microsoft C++ Build Tools; Kaggle / Linux builds it normally |
| FAISS CPU, not FAISS GPU | Gallery searches take milliseconds on CPU; FAISS GPU has no Windows pip package |

### Still open for Phase 1
1. Obtain CUHK-PEDES (Dia), then run the MVP workflow in README section 3.
2. Each member: accept the IndicTrans2 licence on Hugging Face and log in; run
   `scripts/check_translation.py`, then the Indic query generation.
3. Show the demo to the faculty guide (MVP sign-off).
