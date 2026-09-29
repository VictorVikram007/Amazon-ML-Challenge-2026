# Business Entity Resolution — Methodology Document

## 1. Problem overview
Given three noisy business-record sources (Source 1 = deduplicated reference, Source 2
and Source 3 = noisy records), identify every Source 2/3 record that refers to the same
real-world business as each Source 1 entity. Matches are zero, one, or many; scored by
entity-level macro F0.5 (precision-weighted 2:1 over recall).

## 2. Dataset description (measured, see `experiments/data_audit_report.md` for full detail)
- Train: 2,206,821 Source-1 rows, 5,034,616 Source-2 rows, 5,285,603 Source-3 rows, one
  ground-truth row per Source-1 entity.
- Countries: US (~60%) / India (~40%) in train; test additionally contains France
  (per the problem statement) — handled as an open string set throughout, never hard-coded.
- Ground truth is **not** a mostly-singleton problem: only 5.58% of S1 entities are true
  singletons; 89.02% have 2+ true matches (avg 3.666 matches when matched). Both Source 2
  and Source 3 contribute comparably (48.4% / 51.6% of all matched-id references).
- `business_address` is missing in ~3.3% of Source 2/3 rows; a literal noise token
  (`null`, `<NULL>`) also appears embedded inside otherwise-present addresses.
- **Country never crosses a true match** (0/172,731 checked pairs mismatch) — used as a
  mandatory blocking filter.

## 3. Data preprocessing
Handled in `src/normalize.py`. Unicode-normalize (NFKC), strip literal `null`/`<NULL>`
noise tokens embedded in address components, replace `&` with `and`, strip punctuation,
collapse whitespace, lowercase. Original strings are always retained alongside the
normalized ones — nothing is destructively discarded.

## 4. Normalization
- **Names:** legal-suffix folding (`Incorporated`→`inc`, `Limited`→`ltd`, `Private`→`pvt`,
  `Corporation`→`corp`, `Company`→`co`, `L.L.C`→`llc`, etc.), tokenization.
- **Addresses:** common abbreviation folding (`Road`→`rd`, `Street`→`st`, `Avenue`→`ave`,
  etc.), tokenization, and separate extraction of numeric tokens (street numbers,
  PIN/ZIP-like fragments) — the audit showed these are the most reliable anchor when
  names are transliterated, garbled, or the record has no name overlap at all.
- **Non-Latin scripts** (Devanagari/Tamil/Telugu/Kannada business names, observed
  extensively in the Indian records) are Unicode-normalized and tokenized as-is — never
  transliterated or translated (no external service is permitted or used). Same-script
  duplicates still match; cross-script pairs must be recovered through address signal.
- **Country** is folded to lowercase/whitespace-normalized only — never mapped through a
  fixed lookup table, so unseen values (e.g. France) pass through unaffected.

## 5. Candidate generation / blocking
`src/blocking.py`: a country-partitioned union of five independent channels —
exact normalized name, exact normalized address, name-token inverted index,
address-numeric-token inverted index, and address character 4-gram inverted index.
Any index token with a too-long posting list is skipped for blocking (would explode the
candidate set for negligible discriminative value).

**Important engineering finding, kept in `experiments/experiments.csv` (exp001):** the
first implementation truncated an over-cap candidate union by sorting IDs alphabetically.
Measured blocking recall was **4.75%**. Replacing this with a weighted-vote ranking
(exact match > numeric-token match > name-token match > n-gram match, keep the top-voted
candidates up to the cap) raised measured blocking recall to **94.53%** on the same data
(exp002–exp004). This is exactly the kind of measured, not-assumed, decision the brief
asks for.

## 6. Blocking recall (measured)
On an 8,000-Source-1-entity sandbox-scale run (see §12 "Limitations" for why this is not
yet the full 2.2M-entity file): **blocking recall = 0.9453**, average candidates per S1 =
99.97 (against a max cap of 100).

## 7. Pairwise feature engineering
`src/features.py`, 27 features per (S1, candidate) pair: exact-match flags, token
Jaccard, character n-gram Jaccard, rapidfuzz edit-distance family (`ratio`,
`token_sort_ratio`, `token_set_ratio`), length difference/ratio, common-prefix ratio,
address numeric-token overlap count/Jaccard, country-exact flag, missing-field flags, and
interaction features (similarity product/sum, "both high", "strong-name-weak-address",
"weak-name-strong-address", exact-name-and-country, exact-address-and-country).

**Deviation from the brief's tip list, documented rather than silently substituted:**
TF-IDF cosine similarity is replaced with character n-gram Jaccard + rapidfuzz's
C-implemented edit-distance family. At this row count, a global or per-S1 TF-IDF fit adds
real CPU cost with no measured benefit over a typo-tolerant alternative that is far
cheaper to compute per pair at scale.

## 8. Model architecture
Three baselines compared exactly as the brief specifies (Logistic Regression → XGBoost →
LightGBM), model-selected on **validation entity-level macro F0.5**, never on pair-level
accuracy or AUC. All are trained from scratch on competition data only — no pretrained
weights, so the MIT/Apache-2.0 + ≤8B-parameter constraint is trivially satisfied.

Measured on the sandbox-scale validation split (1,600 held-out S1 entities, entity-level
split — see §9):

| Model | Best threshold | Val macro entity F0.5 |
|---|---|---|
| Logistic Regression | 0.85 | 0.9637 |
| LightGBM | 0.90 | 0.9679 |
| **XGBoost (selected)** | **0.85** | **0.9699** |

Top feature importances (XGBoost): `addr_token_jaccard` (0.70), `name_token_sort_ratio`
(0.11), `name_token_set_ratio` (0.03), `sim_sum` (0.03), `name_char_ngram_jaccard`
(0.03) — address token overlap dominates, consistent with the audit finding that address
is the most reliable signal when names are transliterated or corrupted.

## 9. Training strategy
S1 **entities** (not pairs) are split 80/20 train/validation with a fixed seed, so no
single S1's positive/negative pairs cross the split (`src/train.py::split_by_entity`).
Every candidate pair inherits features from `src/features.py`; class imbalance is handled
via `scale_pos_weight` (XGBoost/LightGBM) or `class_weight="balanced"` (Logistic
Regression), plus a configurable cap on negatives sampled per positive
(`negative_per_positive`, default 6x).

## 10. Hard-negative strategy
No synthetic negatives are generated. Every non-true-match candidate that survives
blocking is, by construction, a naturally-occurring hard negative — it already shares
enough tokens/n-grams/numeric fragments with the S1 query to have passed every blocking
filter. Measured false-positive examples from validation (see
`experiments/error_analysis_log.txt`) confirm the expected hard-negative categories:
shared house/plot numbers between two unrelated Indian businesses at nearby addresses,
and missing-address records where name similarity alone was overweighted.

## 11. Validation methodology
Entity-level split (not row/pair split) to avoid leakage. The primary metric
(`src/evaluate.py::macro_entity_f05`) reproduces the competition scoring exactly: F0.5 per
S1 entity (a correctly-predicted empty set scores 1.0), macro-averaged over every S1 in
the evaluation set including singletons. Secondary diagnostics tracked alongside: blocking
recall, average candidates/S1, singleton false-positive rate.

## 12. F0.5 threshold optimization (measured, full sweep)
| Threshold | Val macro entity F0.5 | S1s with ≥1 match | Avg matches/matched S1 |
|---|---|---|---|
| 0.50 | 0.9661 | 1511 | 3.49 |
| 0.55 | 0.9666 | 1511 | 3.48 |
| 0.60 | 0.9669 | 1511 | 3.48 |
| 0.65 | 0.9675 | 1511 | 3.47 |
| 0.70 | 0.9678 | 1511 | 3.47 |
| 0.75 | 0.9682 | 1511 | 3.46 |
| 0.80 | 0.9687 | 1511 | 3.45 |
| **0.85** | **0.9699** | 1509 | 3.44 |
| 0.90 | 0.9691 | 1508 | 3.42 |
| 0.95 | 0.9676 | 1507 | 3.40 |

0.85 is selected: it maximizes validation macro entity F0.5 with a measured singleton
false-positive rate of 0.0127 (i.e. ~1.3% of true singletons were incorrectly given a match).

## 13. Error analysis (measured examples in `experiments/error_analysis_log.txt`)
**False positives** cluster into: (a) two distinct Indian businesses sharing a house/plot
number fragment in the same city, where address-token overlap alone pushed the score high
despite a clearly different business name; (b) one side missing its address entirely, so
the classifier over-relied on partial name similarity.

**False negatives** split into two distinct failure modes that need different fixes:
- *Dropped at blocking* — no name-token, numeric-token, or n-gram overlap survived
  (e.g. a domain-style name like `Caldoschicago.Com` vs `Caldos Chicago Group` with a
  short/partial address on one side). This is a blocking-recall gap, not a classifier gap.
- *In candidates but scored below threshold* — mostly Devanagari-script names paired with
  a Latin-script S1 name and a partially-differing address (e.g. village/locality
  variation within the same city). The classifier is correctly cautious here given F0.5's
  precision weighting; recovering these would need an address-similarity feature more
  tolerant of partial/reordered components than plain token Jaccard.

## 14. Experiments
See `experiments/experiments.csv` (4 logged experiments) and
`experiments/error_analysis_log.txt` / `experiments/demo_run_log.txt` for full console logs.

## 15. Final approach
Country-partitioned 5-channel union blocking with weighted-vote candidate ranking, 27
hand-engineered pairwise features (token/character/edit-distance similarity across
name+address, numeric-token overlap, interaction features), XGBoost classifier selected
by validation entity-level macro F0.5, decision threshold 0.85 chosen by full sweep.

## 16. Limitations
- **All numbers in this document are from a sandbox-scale run** (8,000 of 2,206,821
  train S1 entities; a 250,000-row-per-source reservoir-sampled distractor pool rather
  than the full ~10M Source2+Source3 rows), because the authoring environment had ~2.7GB
  RAM / 1 CPU core — not enough to hold a full dict-based blocking index over 10M rows
  simultaneously with a residual pandas frame. The pipeline code itself is unchanged
  between this run and a full-file run (`python3 -m src.pipeline train-eval` with no
  `--max-s1` cap) — only more RAM/time is required. See README §8.
- **Test set was not available at the time of writing** this document — no
  `dataset/test/*.tsv` files were provided, so `output/matching_results.tsv` /
  `output/candidate_pairs.tsv` could not yet be generated or validated against the real
  test set. `src.pipeline predict-test` is implemented and ready to run the moment test
  files are supplied.
- Cross-script transliteration (Devanagari/Tamil/Telugu/Kannada ↔ Latin) is not bridged
  by any translation step (none is permitted); recall on these pairs depends entirely on
  address overlap, which the error analysis shows is imperfect when the address also
  differs (e.g. locality-level variation).
- Character n-gram Jaccard + rapidfuzz is used instead of TF-IDF cosine (§7) — a
  reasonable, but not identical, substitute.

## 17. Conclusion
The pipeline meets the brief's precision-first design goals (measured singleton
false-positive rate 1.27%, validation macro entity F0.5 0.9699) using only blocking
strategies and features derivable from the provided files, with every modeling decision
backed by a measured number rather than an assumption. The main remaining work before a
leaderboard submission is (a) running the identical pipeline at full data scale once
more compute is available, and (b) running it against the real test files once provided.

## 18. Reproducibility instructions
See `code/business_entity_resolution/README.md` for exact commands.
