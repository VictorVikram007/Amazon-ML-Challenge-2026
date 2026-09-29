import sys, random, time
sys.path.insert(0, "code/business_entity_resolution")
from src import config
from src.blocking import BlockingIndex
from src.io import iter_source_chunks, load_ground_truth_map, read_source_full
from src.evaluate import macro_entity_f05, blocking_recall, entity_f05
from src.train import build_pairs_dataframe, split_by_entity, TRAINERS, predict_proba
from src.predict import decide_matches, search_best_threshold
from src.features import FEATURE_NAMES, compute_pair_features
import pandas as pd

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

SEED = 42
N_S1 = 8000
DISTRACTOR_POOL_PER_SOURCE = 250_000
rng = random.Random(SEED)

log("Rebuilding the same sandbox-scale pool as demo_run.py (same seed) ...")
gt_full = load_ground_truth_map(config.PATHS.train_ground_truth)
s1_df = read_source_full(config.PATHS.train_source1)
all_s1_ids = s1_df["entity_id"].tolist()
sample_s1_ids = set(rng.sample(all_s1_ids, N_S1))
s1_df = s1_df[s1_df["entity_id"].isin(sample_s1_ids)]
s1_records = {r.entity_id: (r.business_name, r.business_address or "", r.country) for r in s1_df.itertuples()}
gt = {sid: gt_full[sid] for sid in sample_s1_ids}
needed_true_ids = set()
for m in gt.values(): needed_true_ids.update(m)

def build_pool(path, needed_ids, pool_cap, seed):
    r = random.Random(seed)
    pool = {}
    reservoir_ids = []
    n_seen = 0
    for chunk in iter_source_chunks(path, chunksize=300_000):
        for eid, name, addr, country in zip(chunk["entity_id"], chunk["business_name"],
                                             chunk["business_address"].fillna(""), chunk["country"]):
            n_seen += 1
            if eid in needed_ids:
                pool[eid] = (name, addr, country); continue
            if len(reservoir_ids) < pool_cap:
                reservoir_ids.append(eid); pool[eid] = (name, addr, country)
            else:
                j = r.randint(0, n_seen - 1)
                if j < pool_cap:
                    old = reservoir_ids[j]
                    if old in pool: del pool[old]
                    reservoir_ids[j] = eid; pool[eid] = (name, addr, country)
    return pool

needed_s2 = {i for i in needed_true_ids if i.startswith("S2-")}
needed_s3 = {i for i in needed_true_ids if i.startswith("S3-")}
pool2 = build_pool(config.PATHS.train_source2, needed_s2, DISTRACTOR_POOL_PER_SOURCE, SEED)
pool3 = build_pool(config.PATHS.train_source3, needed_s3, DISTRACTOR_POOL_PER_SOURCE, SEED)
other_records = {**pool2, **pool3}
log(f"pool sizes: s2={len(pool2)} s3={len(pool3)}")

idx = BlockingIndex(config.BLOCKING)
for eid, (name, addr, country) in other_records.items():
    idx.add_record(eid, name, addr, country)

candidates = {s1_id: idx.candidates_for(*s1_records[s1_id]) for s1_id in s1_records}
b_recall, b_avg, b_max = blocking_recall(gt, candidates)
log(f"blocking recall={b_recall:.4f} avg_cand={b_avg:.2f}")

pairs_df = build_pairs_dataframe(s1_records, other_records, candidates, gt, config.MODEL, SEED)
train_ids, val_ids = split_by_entity(list(s1_records.keys()), 0.2, SEED)
train_mask = pairs_df["s1_id"].isin(train_ids)
val_mask = pairs_df["s1_id"].isin(val_ids)
X_train = pairs_df.loc[train_mask, FEATURE_NAMES].values
y_train = pairs_df.loc[train_mask, "label"].values
X_val = pairs_df.loc[val_mask, FEATURE_NAMES].values

log("Training xgboost (selected model) ...")
model = TRAINERS["xgboost"](X_train, y_train)
val_proba = predict_proba(model, X_val)
val_scored = pairs_df.loc[val_mask, ["s1_id", "cand_id"]].copy()
val_scored["proba"] = val_proba

search = search_best_threshold(val_scored, gt, list(val_ids), config.THRESHOLDS.candidates)
log("FULL THRESHOLD SWEEP (validation, xgboost):")
for r in search["all"]:
    log(f"  t={r['threshold']:.2f}  macro_entity_F0.5={r['macro_entity_f05']:.4f}  "
        f"n_s1_with_match={r['n_s1_with_match']}  avg_matches={r['avg_matches_per_matched_s1']:.2f}")

best_t = search["best"]["threshold"]
preds = decide_matches(val_scored, best_t)

# ---- feature importance ----
try:
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    log("TOP FEATURE IMPORTANCES (xgboost):")
    for name, imp in ranked[:10]:
        log(f"  {name}: {imp:.4f}")
except Exception as e:
    log(f"could not extract feature importances: {e}")

# ---- error analysis: false positives & false negatives on validation ----
val_gt = {s: set(gt[s]) for s in val_ids if s in gt}
fp_examples, fn_examples = [], []
for s1_id in val_ids:
    if s1_id not in gt: continue
    true_set = set(gt[s1_id])
    pred_set = set(preds.get(s1_id, []))
    fps = pred_set - true_set
    fns = true_set - pred_set
    for cid in fps:
        if cid in other_records and len(fp_examples) < 12:
            name1, addr1, c1 = s1_records[s1_id]; name2, addr2, c2 = other_records[cid]
            row = val_scored[(val_scored.s1_id == s1_id) & (val_scored.cand_id == cid)]
            prob = float(row["proba"].iloc[0]) if len(row) else None
            fp_examples.append((s1_id, cid, name1, addr1, name2, addr2, prob))
    for cid in fns:
        if cid in other_records and len(fn_examples) < 12:
            name1, addr1, c1 = s1_records[s1_id]; name2, addr2, c2 = other_records[cid]
            in_cand = cid in candidates.get(s1_id, set())
            fn_examples.append((s1_id, cid, name1, addr1, name2, addr2, in_cand))

log(f"\n=== FALSE POSITIVE examples (predicted match, not true) — n={len(fp_examples)} shown ===")
for s1_id, cid, n1, a1, n2, a2, prob in fp_examples:
    log(f"  {s1_id} vs {cid} (proba={prob:.3f}): S1[{n1!r} | {a1!r}]  vs  CAND[{n2!r} | {a2!r}]")

log(f"\n=== FALSE NEGATIVE examples (true match, missed) — n={len(fn_examples)} shown ===")
for s1_id, cid, n1, a1, n2, a2, in_cand in fn_examples:
    reason = "in candidates but scored below threshold" if in_cand else "DROPPED AT BLOCKING STAGE"
    log(f"  {s1_id} vs {cid} [{reason}]: S1[{n1!r} | {a1!r}]  vs  CAND[{n2!r} | {a2!r}]")
