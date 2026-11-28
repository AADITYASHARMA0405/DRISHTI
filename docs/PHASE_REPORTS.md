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
2. Optional: accept the IndicTrans2 licence on Hugging Face and log in, then rerun
   `scripts/make_indic_queries.py --translator indictrans2` to compare it with the NLLB-200 sets used now.
3. Show the demo to the faculty guide (MVP sign-off).

---

## Phase 1 – Execution on real data (RSTPReid)

### Dataset decision
| Option | Access | Decision |
|---|---|---|
| CUHK-PEDES | Only by request to the original authors; Hugging Face copies are unofficial re-uploads | Requested separately; added when access is granted |
| **RSTPReid** | Official release on the dataset's GitHub (Google Drive link) | **Used for Phase 1** |

RSTPReid (Zhu et al., ACM MM 2021): 20,505 images of 4,101 persons from 15 cameras, two captions per image.
The downloaded copy matches the published split exactly: 3,701 / 200 / 200 identities; the test split has
1,000 images and 2,000 captions. Published reference (ViT-B/16): **TBPS-CLIP 61.95 R1 / 48.26 mAP**,
IRRA 60.20 / 47.17, so AC1 asks for a baseline of about 60 R1.

### AC2 – frozen noisy test sets
* Before freezing the sets, a data-driven check found the parser missed material words between colour and
  garment ("black **down** jacket" 9,707 times, "brown **leather** shoes", ...) and nouns such as overcoat,
  hair, schoolbag and sweatpants. After adding them, the parser finds **3.9 attributes per caption** (was
  3.3), 72% of them with a colour.
* 12 noisy sets (omission / vagueness / commission / mixed × severity 1–3) of 2,000 queries each; 97.9–99.9%
  of queries are changed; mean corrupted attributes ≈ severity.
* Reproducible: generated twice, same checksum **`cfa305ad54858e86`** (seed 2026). **Done.**

### Zero-shot CLIP ViT-B/16 (no training) – the "before" reference

| Query set | R1 | R5 | R10 | mAP |
|---|---|---|---|---|
| clean | 14.15 | 32.45 | 44.30 | 10.26 |
| omission s1 / s2 / s3 | 12.50 / 10.80 / 8.65 | 28.65 / 25.90 / 22.10 | 40.75 / 36.50 / 32.40 | 9.10 / 7.99 / 6.66 |
| vagueness s1 / s2 / s3 | 14.10 / 12.65 / 11.90 | 30.05 / 28.95 / 26.75 | 42.00 / 40.15 / 39.45 | 9.84 / 9.23 / 8.68 |
| commission s1 / s2 / s3 | 11.20 / 9.30 / 7.05 | 29.35 / 25.05 / 21.60 | 40.05 / 35.80 / 31.50 | 8.88 / 7.69 / 6.51 |
| mixed s1 / s2 / s3 | 12.40 / 10.50 / 8.40 | 28.30 / 25.45 / 23.65 | 40.25 / 36.65 / 33.35 | 9.24 / 7.98 / 6.85 |

Three confident mistakes halve Rank-1 (14.15 → 7.05); vagueness hurts least. This is the gap the robust
model must close.

### Problems found while running, and fixes

| Problem | Effect | Fix | Verified by |
|---|---|---|---|
| DataLoader workers kept their own copy of the dataset | The noise curriculum would silently stay at 0% with `num_workers > 0` | Curriculum state moved to a shared-memory tensor that all workers read | `tests/test_curriculum_workers.py` (2 and persistent workers) |
| Recreating 4 workers each epoch on Windows | 3–4 extra minutes per epoch | Persistent workers (safe after the fix above) | Same test |
| Laptop slept overnight; session end killed training | First baseline run lost ~12 hours and stopped in epoch 3 | Training asks Windows to stay awake while it runs; full resume (optimizer, scheduler, scaler, RNG) every epoch; `--stop-after N`; jobs started outside the chat session (`scripts/launch_detached.ps1`) | `tests/test_resume.py`: stopped-and-resumed run matches an uninterrupted run |
| Run script had no model-cache path | Offline run could not find CLIP | `HF_HOME` passed by the launcher | Run started |
| Modern Standby still slept the laptop ("Idle timeout", then "Battery drain budget exceeded" once unplugged) | Second baseline attempt stopped | Keep-awake also asks for the display (`ES_DISPLAY_REQUIRED`, needed on Modern Standby); training pauses itself on battery and continues when the charger is back | `tests/test_power.py` |
| Intel Fortran runtime inside PyTorch's MKL aborted on a console close / Ctrl-C (`forrtl: error (200)`) | Whole job killed with no Python traceback | Launcher sets `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` | No further forrtl aborts |
| The assistant's shell runs inside a Windows job object | Jobs it started died with the session | `scripts/_breakaway.ps1` starts the job with `CREATE_BREAKAWAY_FROM_JOB` | Training survived session restarts |
| Training stopped silently in epoch 3 (23:37, no traceback, no crash event) with its minimised console open | ~5 hours of epoch 3 lost; resume restarts from epoch 2 | The detached console is now hidden, so it cannot be closed by accident; `scripts/stop_training.ps1` is the way to stop it | Next launch |
| Hindi/Kannada translation (NLLB on the GPU) ran next to training | Training slowed from ≈ 1.8 to 15–25 s/step (6 GB GPU shared) | Never run a second GPU job during training; translation is finished now | – |

### AC5 – Hindi and Kannada queries (translate-then-search)
IndicTrans2 is licence-gated on Hugging Face, so `drishti/lang/translate.py` falls back to **NLLB-200
distilled 600M** (open licence) when IndicTrans2 cannot be loaded (`tests/test_translate_fallback.py`).
`scripts/make_indic_queries.py --translator nllb` turned all 2,000 RSTPReid test captions into eight sets:

| Set | What the model receives |
|---|---|
| `hi`, `kn` | Native-script Hindi / Kannada, fed straight to the English encoder (no-translation reference) |
| `hi_rom`, `kn_rom` | The same, romanized (ITRANS) – how people often type on phones |
| `hi_t2s`, `kn_t2s` | Native script translated back to English, then searched (the MVP pipeline) |
| `hi_rom_t2s`, `kn_rom_t2s` | Romanized text → native script → English → search |

How much of each caption survives the round trip (attributes found by the noise parser in the English
caption that are still present after en → L → en; `outputs/t2s_attribute_retention.txt`):

| Set | Garment nouns kept | Colours kept |
|---|---|---|
| hi_t2s | 75.2% | 81.4% |
| hi_rom_t2s | 75.2% | 81.5% |
| kn_t2s | 70.2% | 76.9% |
| kn_rom_t2s | 69.0% | 76.5% |

Findings:
* **The translator, not the language, is the first bottleneck.** A quarter (Hindi) to almost a third
  (Kannada) of garment words and a fifth of colours are lost *before* retrieval starts. Typical losses:
  trousers → "panties", kurta → "robe", dupatta → "sweater", saree → "dressed in red" (Kannada),
  sandals → "shoes" (Hindi). Indian garment terms are hit hardest – the motivation for Phase 2's
  attire-aware vocabulary and an Indic-capable text encoder.
* **Romanization is almost lossless**: ITRANS round trip gives the same English as native script for 90.6%
  of Hindi queries, so romanized input costs little extra once it is converted back.
* Retrieval scores for these sets are computed for every trained model by `scripts/run_phase1.sh`
  (`outputs/results_rstpreid_indic.csv`, `outputs/summary_rstpreid_indic.md`) and are added below as the
  runs finish.

### AC7 – reproducible from a fresh clone
Fresh `git clone` of `phase-1-mvp` (commit 857022f) into an empty folder: the noisy sets regenerate with the
same checksum `cfa305ad54858e86`, zero-shot evaluation gives the same numbers (R1 14.15, mAP 10.26), and the
full test suite passes there on CPU (**35 passed**; 37 min on battery, almost all of it the end-to-end smoke
test). **Done.**

### Runs (in progress)
`scripts/run_phase1.sh 1 2 3`: baseline and robust model × seeds 1–3 on the RTX 4050 (batch 64, gradient
checkpointing, ≈ 1.47 s/step, 578 steps/epoch, 10 epochs ≈ 2.5 h per run), each followed by evaluation on
the clean set and all 12 noisy sets. Results go to `outputs/results_rstpreid.csv` and
`outputs/summary_rstpreid.md`.

Baseline seed 1 so far (validation split, clean queries): epoch 1 R1 34.0 / mAP 27.9, epoch 2 R1 41.3 /
mAP 33.6 (zero-shot: 14.15 R1 on test). Resumes from epoch 2 on the next launch.
