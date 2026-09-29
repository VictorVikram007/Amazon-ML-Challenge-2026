# Amazon ML Challenge 2026: Business Entity Resolution

Link each Source-1 business record to all of its matching records in Source 2 and Source 3
(zero, one or many). Scored with **macro-averaged, per-entity F0.5**. Precision counts
more than recall, and an entity with no true matches scores 1.0 only if nothing is predicted.

**Best leaderboard score: 0.9738** (up from a 0.426 exact-match baseline).
It runs end to end on a 16 GB laptop over roughly 26M records, using polars, rapidfuzz and LightGBM.

## Pipeline (`work/`)

| Step | Files | What it does |
|---|---|---|
| Normalize | `norm.py`, `prep.py` | Cleans case, accents, punctuation and "null" noise. Transliterates Indian scripts (Devanagari, Bengali, Telugu, Tamil, Kannada, Gurmukhi) into sound-alike Latin keys. Extracts whole address codes (`8-2-293/82/A/728` → `8229382a728`) and a filler-free core name. |
| Candidate search | `block.py` | For every S2/S3 record: IDF-weighted overlap on rare words, order-free word pairs, name × address pairs, sound-alike keys, address codes and core name. Takes the top 50, re-ranks them by fuzzy name + address similarity, and keeps the top 8. |
| Pair features | `feats.py`, `featx.py` | Fuzzy name, address and sound-alike similarities, blocking-context features, and **decoy-detection features**: house-number edit distance and gap, and name or street words with no close match. Filler words and OCR-style typos ("5ervices") are ignored. |
| Stage 1 model | `train_model.py` | LightGBM, cross-fitted by S1 so the validation score stays honest. |
| Stage 2 group check | `stage2.py`, `train2.py` | Compares each record with the other records confidently matched to the same S1. |
| Assignment | `predict_test.py`, `write_thr.py` | Each S2/S3 record goes to **at most one** S1, and only above a confidence threshold. |
| Evaluation | `metric.py` | Exact competition metric. |

Run order: `prep.py` → `block.py` → `feats.py` → `featx.py` → `train_model.py` → `stage2.py` → `train2.py`,
then for test `run_test.sh` (or `predict_test.py`). `run_retrain.sh` retrains with 19% of train S1s hidden
so the training data has the same decoy rate as test.

## Key findings

- **Each S2/S3 record belongs to at most one S1** (7.64M train matches, all distinct). Enforcing this was the largest single precision gain.
- **About a quarter of S2/S3 records are planted decoys:** a real entity with the house number nudged (18514 → 18516), one name word swapped, or a different street.
- **Test has about twice train's decoy rate** (about 2.3 decoys per S1 versus 1.2). This explains the train-to-test gap and why a stricter threshold helped on test.
- **The metric is not compressed:** an empty submission scores 0.056, and exact matching scores 0.426.
- **No train/test leakage:** 0% name + address overlap between splits, and IDs and row order carry no signal.

## Score progression (leaderboard)

| Version | Score | Change |
|---|---|---|
| Exact name/address union | 0.426 | baseline |
| v1 | 0.947 | rare-word + word-pair blocking, LightGBM, one-S1-per-record assignment |
| v2 | 0.952 | stage-2 group check, Indian-script transliteration |
| v3 | 0.9696 | top-50 re-ranking, decoy-detection features |
| v4 | 0.97093 | address codes, core-name key for no-address records |
| v6 (threshold 0.85) | **0.9738** | retrained at test's decoy rate |

## Remaining errors (test-like train)

The biggest remaining loss is true matches that carry the same corruptions as decoys: a nudged number or a
made-up brand name. After that come true S1s that never become candidates (Indian-script names, records with
no address), and no-address records whose name matches several S1s.

## Not included

The challenge dataset, generated intermediates, and submission files (see `.gitignore`).
`code/business_entity_resolution/` is an earlier baseline implementation.
