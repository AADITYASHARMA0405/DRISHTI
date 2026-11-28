#!/usr/bin/env bash
# Phase 1 (MVP) runs on RSTPReid: baseline and robust model for the given seeds, then evaluation of every
# checkpoint on all frozen query sets and the summary table.
#
#   bash scripts/run_phase1.sh 1          # one seed
#   bash scripts/run_phase1.sh 1 2 3      # all three seeds
#
# Safe to re-run at any time: finished runs are skipped and interrupted runs continue from their last
# completed epoch (train.py --resume). Runs offline from the Hugging Face cache, so download CLIP once first
# (any eval.py run does it) and set HF_HOME if the cache is not in the default place.
# To start it so it keeps running after you close the terminal, use scripts/launch_detached.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
RESULTS=outputs/results_rstpreid.csv
INDIC_RESULTS=outputs/results_rstpreid_indic.csv
mkdir -p outputs/logs

for SEED in "$@"; do
  for MODEL in baseline robust; do
    RUN=rstp_${MODEL}_seed${SEED}
    CFG=configs/local/rtx4050_rstp_${MODEL}.yaml
    if [ ! -f "outputs/${RUN}/.done" ]; then
      echo "[phase1] training ${RUN}  $(date)"
      "$PY" train.py --config "$CFG" --seed "$SEED" --resume >> "outputs/logs/${RUN}.log" 2>&1
    fi
    if [ ! -f "outputs/${RUN}/gallery.npz" ]; then
      echo "[phase1] evaluating ${RUN}  $(date)"
      "$PY" eval.py --config "$CFG" --checkpoint "outputs/${RUN}/best.pt" --run "$RUN" \
          --query-sets "query_sets/rstpreid/*_s*.json" --results "$RESULTS" \
          --save-gallery "outputs/${RUN}/gallery.npz" >> "outputs/logs/${RUN}.log" 2>&1
    fi
    # Hindi / Kannada queries (scripts/make_indic_queries.py): translate-then-search (*t2s) and, as the
    # no-translation reference, the raw native-script and romanized text fed straight to the English encoder.
    if ls query_sets/rstpreid/*t2s.json > /dev/null 2>&1 && [ ! -f "outputs/${RUN}/.indic_done" ]; then
      echo "[phase1] evaluating ${RUN} on Hindi/Kannada queries  $(date)"
      "$PY" eval.py --config "$CFG" --checkpoint "outputs/${RUN}/best.pt" --run "$RUN" \
          --query-sets query_sets/rstpreid/clean.json "query_sets/rstpreid/[hk][in]*.json" --results "$INDIC_RESULTS" \
          >> "outputs/logs/${RUN}.log" 2>&1 && touch "outputs/${RUN}/.indic_done"
    fi
  done
done
"$PY" scripts/summarise_results.py --results "$RESULTS" --out outputs/summary_rstpreid.md > /dev/null
if [ -f "$INDIC_RESULTS" ]; then
  "$PY" scripts/summarise_results.py --results "$INDIC_RESULTS" --out outputs/summary_rstpreid_indic.md > /dev/null
fi
echo "[phase1] done  $(date)"
