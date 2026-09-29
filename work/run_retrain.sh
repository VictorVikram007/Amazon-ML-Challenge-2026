#!/bin/bash
# Retrain with 19% of train S1 hidden (test-like decoy rate), then re-score test.
cd /d/Downloads/amazon_ml_challenge_submission
PY=.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
set -x
if [ ! -f work/train/_p.parquet ] && [ -z "$(ls work/train/feat 2>/dev/null)" ]; then
for c in india us; do
  if [ ! -f work/train/cand_parts/$c.done ]; then
    POLARS_MAX_THREADS=8 $PY work/prep.py train $c 19 || exit 1
    for i in 1 2 3; do POLARS_MAX_THREADS=8 $PY work/block.py train 50 50 8 4 nomerge; [ -f work/train/cand_parts/$c.done ] && break; done
    [ -f work/train/cand_parts/$c.done ] || exit 1
    rm -rf work/train/tok_* work/train/xtok_*
  fi
done
[ -f work/train/cand.parquet ] || [ -f work/train/_p.parquet ] || { POLARS_MAX_THREADS=4 $PY -c "import polars as pl; pl.scan_parquet('work/train/cand_parts/*.parquet').sink_parquet('work/train/cand.parquet')" && rm -rf work/train/cand_parts; }
fi
#recall skipped on resume
for i in 1 2 3 4; do POLARS_MAX_THREADS=4 $PY work/feats.py train 5; [ ! -f work/train/_p.parquet ] && [ -n "$(ls work/train/feat)" ] && break; done
for i in 1 2 3; do POLARS_MAX_THREADS=4 $PY work/featx.py train; done
$PY -c "import polars as pl,glob,sys; fs=glob.glob('work/train/feat/*.parquet'); n=sum('ux_o' in pl.read_parquet_schema(f) for f in fs); print('featx',len(fs),n); sys.exit(0 if n==len(fs) and n>0 else 1)" || exit 1
POLARS_MAX_THREADS=8 $PY work/train_model.py || exit 1
for i in 1 2 3; do POLARS_MAX_THREADS=4 $PY work/stage2.py train oof.parquet; [ -f work/train/s2.parquet ] && break; done
POLARS_MAX_THREADS=8 $PY work/train2.py || exit 1
for i in 1 2; do POLARS_MAX_THREADS=6 $PY work/predict_test.py 0.65 submission_v6 && break; done
$PY utils/validate_submission.py --matching submission_v6/matching_results.tsv --candidate submission_v6/candidate_pairs.tsv --test-dir dataset/test
