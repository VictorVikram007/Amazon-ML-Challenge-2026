"""
Sandbox-scale end-to-end validation of the real pipeline code (not a toy
reimplementation — imports directly from code/business_entity_resolution/src).

Because this sandbox has ~2.7GB RAM / 1 CPU core and the full S2+S3 files are
~10M rows, we cannot hold a full-file blocking index in memory here. Instead:

  1. Sample N_S1 Source-1 entities.
  2. Look up their true matches from ground truth (this does NOT bias the
     blocking-recall measurement — it only guarantees the true match ROWS are
     available in the record pool; blocking still has to actually retrieve
     them via token/n-gram overlap, exactly as it would on the full file).
  3. Reservoir-sample a large pool of OTHER S2/S3 rows as realistic
     distractors (same country mix as the full file), so negative candidates
     found by blocking are genuine confusors, not synthetic.
  4. Run the real BlockingIndex / feature / train / evaluate / predict code
     over this pool.

This produces real, honestly-measured numbers at reduced scale. The same
`src.pipeline.run_train_eval` runs unchanged on the full files given more
RAM/time (e.g. the user's own machine) — see README.md.
"""
import random
import sys
import time
sys.path.insert(0, "code/business_entity_resolution")

from src import config
from src.blocking import BlockingIndex
from src.io import iter_source_chunks, load_ground_truth_map, read_source_full
from src.evaluate import macro_entity_f05, blocking_recall, singleton_false_positive_rate
from src.train import build_pairs_dataframe, split_by_entity, TRAINERS, predict_proba
from src.predict import decide_matches, search_best_threshold
from src.features import FEATURE_NAMES
import pandas as pd

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

SEED = 42
N_S1 = 8000
DISTRACTOR_POOL_PER_SOURCE = 250_000

rng = random.Random(SEED)

log("Loading ground truth ...")
gt_full = load_ground_truth_map(config.PATHS.train_ground_truth)

log("Loading Source1 and sampling S1 entities ...")
s1_df = read_source_full(config.PATHS.train_source1)
all_s1_ids = s1_df["entity_id"].tolist()
sample_s1_ids = set(rng.sample(all_s1_ids, N_S1))
s1_df = s1_df[s1_df["entity_id"].isin(sample_s1_ids)]
s1_records = {r.entity_id: (r.business_name, r.business_address or "", r.country)
              for r in s1_df.itertuples()}
gt = {sid: gt_full[sid] for sid in sample_s1_ids}
log(f"  sampled {len(s1_records)} S1 entities; "
    f"{sum(1 for v in gt.values() if len(v)==0)} singletons, "
    f"{sum(1 for v in gt.values() if len(v)>0)} with >=1 true match")

needed_true_ids = set()
for m in gt.values():
    needed_true_ids.update(m)
log(f"  {len(needed_true_ids)} true-match ids to guarantee inclusion")

# Reservoir sample distractors + guaranteed inclusion of true-match rows.
def build_pool(path, needed_ids, pool_cap, seed):
    r = random.Random(seed)
    pool = {}  # entity_id -> (name, addr, country)
    reservoir_ids = []
    n_seen = 0
    for chunk in iter_source_chunks(path, chunksize=300_000):
        for eid, name, addr, country in zip(chunk["entity_id"], chunk["business_name"],
                                             chunk["business_address"].fillna(""), chunk["country"]):
            is_needed = eid in needed_ids
            n_seen += 1
            if is_needed:
                pool[eid] = (name, addr, country)
                continue
            # reservoir sampling for the distractor pool
            if len(reservoir_ids) < pool_cap:
                reservoir_ids.append(eid)
                pool[eid] = (name, addr, country)
            else:
                j = r.randint(0, n_seen - 1)
                if j < pool_cap:
                    old = reservoir_ids[j]
                    if old in pool:
                        del pool[old]
                    reservoir_ids[j] = eid
                    pool[eid] = (name, addr, country)
    return pool

log(f"Building record pool from Source2 (cap {DISTRACTOR_POOL_PER_SOURCE} distractors + true matches) ...")
needed_s2 = {i for i in needed_true_ids if i.startswith("S2-")}
pool2 = build_pool(config.PATHS.train_source2, needed_s2, DISTRACTOR_POOL_PER_SOURCE, SEED)
log(f"  pool2 size = {len(pool2)}")

log(f"Building record pool from Source3 (cap {DISTRACTOR_POOL_PER_SOURCE} distractors + true matches) ...")
needed_s3 = {i for i in needed_true_ids if i.startswith("S3-")}
pool3 = build_pool(config.PATHS.train_source3, needed_s3, DISTRACTOR_POOL_PER_SOURCE, SEED)
log(f"  pool3 size = {len(pool3)}")

missing_true = (needed_true_ids - set(pool2.keys()) - set(pool3.keys()))
log(f"  true-match ids missing from pools (should be 0): {len(missing_true)}")

other_records = {**pool2, **pool3}

log("Building blocking index over the pool ...")
idx = BlockingIndex(config.BLOCKING)
for eid, (name, addr, country) in other_records.items():
    idx.add_record(eid, name, addr, country)
log(f"  index stats: { {k: v for k,v in idx.stats().items() if k!='records_per_country'} }, "
    f"per-country={idx.stats()['records_per_country']}")

log("Computing candidates for every sampled S1 ...")
candidates = {}
for s1_id, (name, addr, country) in s1_records.items():
    candidates[s1_id] = idx.candidates_for(name, addr, country)
avg_cand = sum(len(v) for v in candidates.values()) / len(candidates)
max_cand = max(len(v) for v in candidates.values())
log(f"  avg candidates/S1 = {avg_cand:.2f}, max = {max_cand}")

b_recall, b_avg, b_max = blocking_recall(gt, candidates)
log(f"  BLOCKING RECALL (against this pool) = {b_recall:.4f}")

log("Building labeled pair table ...")
pairs_df = build_pairs_dataframe(s1_records, other_records, candidates, gt, config.MODEL, SEED)
log(f"  {len(pairs_df)} pairs, positive rate = {pairs_df['label'].mean():.4f}")

train_ids, val_ids = split_by_entity(list(s1_records.keys()), 0.2, SEED)
train_mask = pairs_df["s1_id"].isin(train_ids)
val_mask = pairs_df["s1_id"].isin(val_ids)
X_train = pairs_df.loc[train_mask, FEATURE_NAMES].values
y_train = pairs_df.loc[train_mask, "label"].values
X_val = pairs_df.loc[val_mask, FEATURE_NAMES].values
y_val = pairs_df.loc[val_mask, "label"].values
log(f"  train pairs={len(X_train)} (pos={int(y_train.sum())}), val pairs={len(X_val)} (pos={int(y_val.sum())})")

results = {}
for name, trainer in TRAINERS.items():
    log(f"Training {name} ...")
    t0 = time.time()
    model = trainer(X_train, y_train)
    log(f"  trained in {time.time()-t0:.1f}s")
    val_proba = predict_proba(model, X_val)
    val_scored = pairs_df.loc[val_mask, ["s1_id", "cand_id"]].copy()
    val_scored["proba"] = val_proba
    search = search_best_threshold(val_scored, gt, list(val_ids), config.THRESHOLDS.candidates)
    best = search["best"]
    log(f"  {name}: best_threshold={best['threshold']:.2f} "
        f"val_macro_entity_F0.5={best['macro_entity_f05']:.4f} "
        f"n_s1_with_match={best['n_s1_with_match']} avg_matches={best['avg_matches_per_matched_s1']:.2f}")
    results[name] = {"model": model, "search": search, "val_scored": val_scored}

best_name = max(results, key=lambda n: results[n]["search"]["best"]["macro_entity_f05"])
best_entry = results[best_name]
best_threshold = best_entry["search"]["best"]["threshold"]
log(f"SELECTED: {best_name}, threshold={best_threshold}, "
    f"val macro entity F0.5={best_entry['search']['best']['macro_entity_f05']:.4f}")

preds = decide_matches(best_entry["val_scored"], best_threshold)
val_gt = {s: gt[s] for s in val_ids if s in gt}
sfpr = singleton_false_positive_rate(val_gt, preds)
log(f"  singleton false-positive rate on validation = {sfpr:.4f}")

import joblib, os
os.makedirs(config.PATHS.models_dir, exist_ok=True)
joblib.dump(best_entry["model"], os.path.join(config.PATHS.models_dir, f"{best_name}_demo.joblib"))

print("\n=== SUMMARY (measured, sandbox-scale demo) ===")
for n, r in results.items():
    print(f"{n}: {r['search']['best']}")
print(f"blocking_recall={b_recall:.4f} avg_candidates_per_s1={avg_cand:.2f}")
print(f"selected_model={best_name} threshold={best_threshold} singleton_fp_rate={sfpr:.4f}")
