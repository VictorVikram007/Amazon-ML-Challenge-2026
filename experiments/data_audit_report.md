# Phase 0 — Data Audit Report (measured, train split only; test files not yet provided)

## 1. Row counts
| File | Rows |
|---|---|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 (one row per S1 entity — 1:1 with source1) |

## 2. Columns
All three source files: `entity_id, business_name, business_address, country`.
Ground truth: `source1_entity_id, matched_entity_ids`.

## 3. Missing values (measured)
- source1: 0 missing in any column.
- source2: `business_address` missing in 168,967 rows (3.36%). No other missing.
- source3: `business_address` missing in 175,916 rows (3.33%). No other missing.
- Additionally, a literal string token `null` / `<NULL>` appears **inside** non-missing address
  values (e.g. `"KANSAS CITY, MO, 630 45ND TERRACE, null"`, `"33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA"`).
  This is noise, not a missing-value flag from pandas' perspective — normalization must strip it.

## 4. Unique values / duplicates
- entity_id prefixes are clean: source1 is 100% `S1-`, source2 100% `S2-`, source3 100% `S3-`.
- Zero duplicate `entity_id` values within any of the three source files.
- Zero duplicate `source1_entity_id` rows in ground truth (each S1 appears exactly once).

## 5. Country distribution (train only)
- source1: US 1,323,633 (60.0%), India 883,188 (40.0%)
- source2: US 3,016,817 (59.9%), India 2,017,799 (40.1%)
- source3: US 3,170,056 (60.0%), India 2,115,547 (40.0%)
- Only US/India appear in train, matching the spec. Test is confirmed to add **France** —
  the pipeline must never hard-code a 2-value country set.
- **Country integrity check (measured on a 50k-row GT sample, 172,731 checked pairs):
  0 country mismatches (0.0000%) between an S1 entity and any of its true matches.**
  Country is a safe, lossless *mandatory* blocking filter — no true match crosses country.

## 6. Business-name length stats (chars)
| Source | mean | std | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| S1 | 24.03 | 7.74 | 24.0 | 37.0 | 3 | 78 |
| S2 | 25.10 | 8.89 | 25.0 | 40.0 | 2 | 85 |
| S3 | 25.20 | 9.49 | 25.0 | 42.0 | 2 | 83 |

## 7. Business-address length stats (chars, non-missing only)
| Source | mean | std | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| S1 | 52.07 | 25.33 | 41.0 | 102.0 | 13 | 205 |
| S2 | 47.83 | 23.70 | 37.0 | 97.0 | 10 | 209 |
| S3 | 48.32 | 20.18 | 42.0 | 92.0 | 6 | 204 |

## 8. Ground-truth match-count distribution (all 2,206,821 S1 entities, exact counts)
- Zero matches (singletons): 123,247 (5.58%)
- One match: 119,157 (5.40%)
- Multiple matches: 1,964,417 (**89.02%**)
- Average matches per S1 (including zero-match): 3.461
- Average matches per S1 (matched entities only): 3.666
- Histogram of match count k: {0: 123247, 1: 119157, 2: 375212, 3: 530841, 4: 484115,
  5: 321957, 6: 164868, 7: 63968, 8: 18680, 9: 4205, 10: 534, 11: 37}
- Of 7,638,365 total matched-id references: 48.4% are S2 IDs, 51.6% are S3 IDs — both
  sources contribute comparably; the pipeline must search both for every S1.

**Implication for strategy:** this is NOT a mostly-singleton dedup problem (only 5.6% singletons).
The overwhelming majority of S1 entities have 2–6 true matches. Recall of the *candidate
generation* stage is the binding constraint (a missed candidate can never be recovered by the
classifier), while the final decision layer still must be precision-heavy per F0.5.

## 9. Real noisy-variation examples (from 20 inspected true match groups)
Concrete patterns actually observed in the data (not assumed):
- **Script/transliteration**: the same Indian business appears with a Devanagari/Tamil/Telugu/
  Kannada-script name in one record and a Latin-script name in another
  (e.g. `राज इन्वेस्टमेंट्स एलएलपी` / `Raj Investments LLP`). No translation is available or
  permitted — for these pairs, name similarity is close to useless and **address similarity is
  the only usable signal**.
- **Character-level typos/corruption** in names: `Payne Enterprises` → `Payne Énterprises`,
  `PAYNE-ENRTPRMISES`, `Payne Etrepndiels`; `Dahlia Power Reliable` → `Dahlia Ponr Reliable`.
- **Token reordering / injected generic suffix tokens**: `AP Hospitality Inc` →
  `AP Inc Hospitality`, `AP AP Hospitality Incorporated`; random appended words like
  `Center`, `Services`, `Partners`, `Company` are injected into otherwise-matching names.
- **Completely unrelated/garbled name, matched purely by address**: e.g. `Dréxkor` and
  `maurewilliamscolombier.com` both truly match `Maure Williams Colombier Inc` — the name
  channel gives zero signal here; address (`85 Wa[y/n]ne Avenue, Ticonderoga ...`) is the only
  anchor. Domain-name-style business names (`wilfordhancock.com`) also occur.
- **Address noise**: component reordering (`"KANSAS CITY, MO, 630 45ND TERRACE, null"` vs
  `"630 45th Terrace, Kansas City, MO"`), street/city typos (`Wanye`→`Wayne`,
  `Ticonderoga Townshiip`→`Ticonderoga`), state written as full name vs abbreviation
  (`New York` vs `NY`, `Illinois` vs `IL`), literal `null`/`<NULL>` noise tokens, and missing
  address entirely for some records in an otherwise-matching group.
- Multiple true matches per S1 can individually have missing/garbled addresses — the group as
  a whole is recoverable via blocking, but each pairwise decision must be made on whatever
  signal that specific pair actually has.

## 10. Leakage risks
- No shared IDs between S1/S2/S3 ID spaces (disjoint prefixes) — no trivial ID leakage.
- Ground truth is 1:1 with source1 rows and was generated independently of any derived
  features, so no obvious column-level leakage found in the raw files.
- The real leakage risk is **structural**: a naive random row-level train/val split does not
  leak here because each row is one S1 entity with its own independent candidate pool, but
  candidate *pairs* generated from the same S1 must stay together in one split (no splitting a
  single S1's positive/negative pairs across train and validation) — enforced by splitting on
  `source1_entity_id`, not on pairs.

## 11. Train/test structural differences
- Test is not yet uploaded. Per the problem statement: test adds a third country, **France**,
  absent from train. The normalization/blocking/feature code must not assume a fixed country
  vocabulary, and country-dependent address parsing (if any) must degrade gracefully to
  generic token-based handling for unseen countries.

## 12. Resource note
This sandbox has ~2.7 GB free RAM and 1 CPU core. Source2+Source3 combined are ~1GB raw text;
loading both simultaneously as pandas string frames risks OOM. The pipeline is written to
avoid full Cartesian products and to stream files in chunks through dict-based inverted
indexes rather than holding multi-million-row DataFrames in memory at once. End-to-end
validation in this session is run on a stratified subset of S1 entities for speed/memory
safety; the same code operates on the full files unchanged given more RAM/time (e.g. on the
user's own machine), by just pointing config at the full paths.
