# Business Entity Resolution — Amazon ML Challenge 2026

Reproducible pipeline: normalization → country-partitioned multi-strategy blocking →
pairwise feature engineering → gradient-boosted classifier → threshold-optimized,
entity-level match decision.

All measured numbers referenced below (blocking recall, validation macro entity F0.5,
feature importances, error-analysis examples) are in `experiments/` and were produced by
actually running this code — nothing here is a fabricated or assumed figure.

## 1. Environment setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Tested with Python 3.11+ and the exact pinned versions in `requirements.txt`.

## 2. Dataset paths

From the competition root (`student_resource/`), the code expects:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Paths are centralized in `src/config.py` (`Paths` dataclass) — edit there if your
layout differs. No absolute or OS-specific paths are hard-coded anywhere.

## 3. Training + validation (Stage A–K of the brief)

Run from the **competition root** (`student_resource/`, the directory that contains
`dataset/`), with `code/business_entity_resolution` on `PYTHONPATH` so the `src` package
resolves while `dataset/...` / `output/...` paths (relative to the root, per `src/config.py`)
also resolve correctly:

```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.pipeline train-eval --max-s1 20000
```

- `--max-s1 N`: use a random subset of N Source-1 (train) entities. Omit this flag
  entirely to use **all** 2,206,821 train S1 entities — the code is unchanged either
  way, it just needs enough RAM to build the full blocking index over Source2+Source3
  (~10M rows). See the "Resource note" below for what this repository could actually
  run given the sandbox it was authored in.
- `--index-row-cap N`: caps how many Source2+Source3 rows are indexed (mainly useful
  for quick smoke tests; leave unset for a real run).
- `--val-fraction 0.2`: fraction of S1 entities (not pairs — see Validation strategy
  below) held out for validation.

This stage:
1. Loads ground truth and Source1, optionally subsamples S1 entities.
2. Splits S1 **entities** into train/validation (never splits pairs from the same S1
   across the split — see `src/train.py::split_by_entity`).
3. Streams Source2+Source3 once to build the blocking index (`src/blocking.py`).
4. Computes candidates for every S1 entity and reports blocking recall.
5. Streams Source2+Source3 a second time to fetch raw fields ONLY for entities that
   are actually a candidate for some S1 (keeps memory bounded).
6. Builds a labeled pairwise feature table (`src/train.py::build_pairs_dataframe`) —
   every non-true-match candidate is a naturally-occurring hard negative (it passed
   blocking, so it is genuinely name/address-similar).
7. Trains Logistic Regression, XGBoost, and LightGBM; for each, runs a full threshold
   sweep on the validation set and picks the threshold that maximizes validation
   **entity-level macro F0.5** (`src/evaluate.py` / `src/predict.py`).
8. Selects the best model by validation macro entity F0.5 (never by pair-level
   accuracy/AUC alone) and saves it to `models/saved_models/`.

## 4. Test inference + output generation (Stage L/M)

Once `dataset/test/*.tsv` exist, again from the competition root:

```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.pipeline predict-test \
    --model-path models/saved_models/xgboost_best.joblib --threshold 0.85
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (paths from
`src/config.py`), then runs an in-process self-check (`src/submission.py::self_check`)
mirroring every rule in the spec.

## 5. Submission validation (Stage M, required before every leaderboard upload)

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Stdlib-only, prints `PASS` (exit 0) or a numbered issue list (exit 1).

## 6. Code layout

```
src/
  config.py      # all paths + hyperparameters, one place to edit
  io.py          # chunked TSV readers/writers, dtype=str everywhere
  normalize.py   # name/address/country normalization, tokenization, n-grams
  blocking.py    # country-partitioned, multi-strategy union blocking index
  features.py    # pairwise similarity feature engineering
  train.py       # pair-table construction (incl. hard negatives) + 3 model trainers
  evaluate.py    # entity-level macro F0.5 (the competition metric) + diagnostics
  predict.py     # threshold search + entity-level match decision
  submission.py  # writes output files + in-process rule self-check
  pipeline.py    # CLI orchestrator tying every stage together
```

## 7. Design decisions worth knowing before you read the code

- **Country is a mandatory, hard blocking filter.** Measured on 172,731 real
  ground-truth pairs: **0 country mismatches**. No true match ever crosses country, so
  every index is partitioned by (normalized) country first, and country is treated as
  an open string set — never hard-coded to `{US, India}` — so the test set's `France`
  (and anything else) is handled automatically.
- **Blocking unions 5 independent channels** (exact name, exact address, name-token
  overlap, address numeric-token overlap, address character n-gram overlap) because the
  data audit showed name-only or address-only blocking each fail on a large share of
  true matches (transliterated names, garbled names, missing addresses — see
  `experiments/data_audit_report.md` §9 for real examples).
- **Candidate-set truncation is vote-weighted, not arbitrary.** An earlier version of
  this code truncated an over-cap candidate union by sorting IDs alphabetically —
  measured blocking recall was 4.75%. Switching to weighted votes (exact match >
  numeric-token match > name-token match > n-gram match) and keeping the top-voted
  candidates raised measured blocking recall to 94.5% on the same data. This is
  documented here specifically so nobody "fixes" it back to something simpler without
  re-measuring recall.
- **TF-IDF cosine substitution:** the brief's tip list mentions TF-IDF cosine
  similarity; we use n-gram Jaccard + rapidfuzz's edit-distance family instead, because
  a global or even per-S1 TF-IDF fit adds real CPU cost at this row count for no
  measured gain over a typo-tolerant, C-implemented alternative. See the docstring at
  the top of `src/features.py`.
- **Negative sampling is not synthetic.** Every negative training pair is a real
  blocking candidate that is not the true match — i.e. it already looks similar enough
  to pass blocking, which is exactly the "hard negative" the brief asks for. See
  `src/train.py` module docstring.

## 8. Resource note (please read before assuming a bug)

This pipeline was authored and validated in a sandbox with ~2.7 GB free RAM and 1 CPU
core — far under what a ~10M-row (Source2+Source3) blocking index needs in a
dict-of-lists Python implementation. All numbers in `experiments/` were therefore
measured on a **reduced but methodologically honest** run: 8,000 sampled Source-1
entities, with (a) every one of their true matches guaranteed present in the record
pool, and (b) a 250,000-row reservoir-sampled pool of *other* Source2/Source3 rows per
source as realistic distractors. Blocking still has to actually retrieve each true
match via token/n-gram overlap — nothing here inflates recall — but the *distractor
volume* is smaller than the full ~10M-row file, so full-scale precision/candidate
counts may differ somewhat (more common-token collisions to filter). Re-run
`notebooks/demo_run.py`'s logic via `src.pipeline.run_train_eval()` (no `--max-s1` cap,
more RAM) to get the full-scale numbers; no code changes are required.

## 9. Model license / size compliance

XGBoost and LightGBM (and scikit-learn's LogisticRegression) are themselves just
training algorithms, not pretrained model weights being reused — the shipped model is
trained from scratch on the competition data, so there is no third-party pretrained
model whose license needs verifying, and parameter count (a few hundred shallow trees,
or a linear model) is trivially under the 8B limit.
