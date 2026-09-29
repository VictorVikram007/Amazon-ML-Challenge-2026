#!/bin/bash
# full test pipeline -> submission_v4 (resumable; each step retried)
cd /d/Downloads/amazon_ml_challenge_submission
PY=.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
set -x
#POLARS_MAX_THREADS=8 $PY work/save1.py || exit 1
#rm -rf work/train/feat work/train/cand.parquet work/train/s2.parquet work/train/oof.parquet
#[ -f work/test/so.parquet ] || POLARS_MAX_THREADS=8 $PY work/prep.py test || exit 1
#for i in 1 2 3; do [ -f work/test/cand.parquet ] && break; POLARS_MAX_THREADS=8 $PY work/block.py test 50 50 8 4; done
#[ -f work/test/cand.parquet ] || exit 1
rm -rf work/test/tok_* work/test/xtok_*
for i in 1 2 3 4; do POLARS_MAX_THREADS=4 $PY work/feats.py test 5; [ ! -f work/test/_p.parquet ] && [ -n "$(ls work/test/feat)" ] && break; done
for i in 1 2 3; do POLARS_MAX_THREADS=4 $PY work/featx.py test; done
$PY -c "import polars as pl,glob,sys; fs=glob.glob('work/test/feat/*.parquet'); n=sum('ux_o' in pl.read_parquet_schema(f) for f in fs); print('featx',len(fs),n); sys.exit(0 if n==len(fs) and n>0 else 1)" || exit 1
POLARS_MAX_THREADS=6 $PY work/predict_test.py 0.65 submission_v4 || exit 1
$PY utils/validate_submission.py --matching submission_v4/matching_results.tsv --candidate submission_v4/candidate_pairs.tsv --test-dir dataset/test
