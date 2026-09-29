"""
End-to-end orchestrator. Run as:

    python3 -m src.pipeline train-eval  --max-s1 20000
    python3 -m src.pipeline predict-test

from the `code/business_entity_resolution/` directory (see README.md).

Design for memory safety (Phase 14 of the brief): source2/source3 are only
ever read in two lightweight streaming passes:
  Pass 1 (index build): stream every row, keep only tokens/ids in the
          blocking index (not the raw name/address strings).
  Pass 2 (record fetch): stream again, keep raw (name, address, country)
          ONLY for entity_ids that survived blocking as an actual candidate
          for some S1 (a much smaller set than the full file).
This avoids ever holding all of source2+source3's raw text in memory at once.

`--max-s1` lets this run at a reduced scale on constrained hardware (this
sandbox has ~2.7GB RAM / 1 CPU core); omit it to run over the full file on a
machine with enough memory — no code changes needed.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .blocking import BlockingIndex
from .io import iter_source_chunks, load_ground_truth_map, read_source_full
from .evaluate import macro_entity_f05, blocking_recall, singleton_false_positive_rate
from .train import build_pairs_dataframe, split_by_entity, TRAINERS, predict_proba
from .predict import decide_matches, search_best_threshold
from .features import FEATURE_NAMES
from .submission import write_matching_results, write_candidate_pairs, self_check


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_index_streaming(paths: List[str], cfg: config.BlockingConfig,
                           chunksize: int = config.CHUNK_SIZE,
                           row_cap: int = None) -> BlockingIndex:
    """Pass 1: build the blocking index from one or more source files without
    holding the raw dataframe in memory beyond the current chunk."""
    idx = BlockingIndex(cfg)
    n = 0
    for path in paths:
        for chunk in iter_source_chunks(path, chunksize=chunksize):
            for eid, name, addr, country in zip(
                chunk["entity_id"], chunk["business_name"],
                chunk["business_address"].fillna(""), chunk["country"],
            ):
                idx.add_record(eid, name, addr, country)
                n += 1
                if row_cap and n >= row_cap:
                    log(f"  reached row_cap={row_cap} while indexing {path}")
                    return idx
        log(f"  indexed {path} (cumulative rows: {n})")
    return idx


def fetch_needed_records(paths: List[str], needed_ids: Set[str],
                          chunksize: int = config.CHUNK_SIZE) -> Dict[str, Tuple[str, str, str]]:
    """Pass 2: stream again, keep raw fields only for entity_ids in needed_ids."""
    out: Dict[str, Tuple[str, str, str]] = {}
    remaining = set(needed_ids)
    for path in paths:
        if not remaining:
            break
        for chunk in iter_source_chunks(path, chunksize=chunksize):
            hit = chunk[chunk["entity_id"].isin(remaining)]
            if len(hit):
                for eid, name, addr, country in zip(
                    hit["entity_id"], hit["business_name"],
                    hit["business_address"].fillna(""), hit["country"],
                ):
                    out[eid] = (name, addr, country)
                remaining -= set(hit["entity_id"])
    return out


def run_train_eval(max_s1: int = None, val_fraction: float = 0.2, seed: int = 42,
                    index_row_cap: int = None):
    cfg_paths = config.PATHS
    bcfg = config.BLOCKING
    mcfg = config.MODEL

    log("Loading ground truth ...")
    gt = load_ground_truth_map(cfg_paths.train_ground_truth)
    log(f"  {len(gt)} S1 entities in ground truth")

    log("Loading Source1 (train) ...")
    s1_df = read_source_full(cfg_paths.train_source1)
    s1_ids_all = s1_df["entity_id"].tolist()

    if max_s1:
        rng = random.Random(seed)
        s1_ids_all = rng.sample(s1_ids_all, min(max_s1, len(s1_ids_all)))
    s1_df = s1_df[s1_df["entity_id"].isin(set(s1_ids_all))]
    s1_records = {
        r.entity_id: (r.business_name, r.business_address or "", r.country)
        for r in s1_df.itertuples()
    }
    log(f"  using {len(s1_records)} S1 entities for this run")

    train_ids, val_ids = split_by_entity(list(s1_records.keys()), val_fraction, seed)
    log(f"  split: {len(train_ids)} train S1 / {len(val_ids)} val S1 (entity-level split, no leakage)")

    log("Pass 1/2: building blocking index over Source2 + Source3 ...")
    idx = build_index_streaming([cfg_paths.train_source2, cfg_paths.train_source3], bcfg,
                                 row_cap=index_row_cap)
    log(f"  index stats: {json.dumps(idx.stats()['records_per_country'])}")

    log("Computing candidates for every S1 in this run ...")
    candidates: Dict[str, Set[str]] = {}
    for s1_id, (name, addr, country) in s1_records.items():
        candidates[s1_id] = idx.candidates_for(name, addr, country)
    avg_cand = sum(len(v) for v in candidates.values()) / len(candidates)
    log(f"  avg candidates/S1 = {avg_cand:.2f}")

    b_recall, b_avg, b_max = blocking_recall(gt, candidates)
    log(f"  BLOCKING RECALL = {b_recall:.4f} (avg candidates={b_avg:.2f}, max={b_max})")

    log("Pass 2/2: fetching raw fields for candidate entities only ...")
    needed_ids = set()
    for c in candidates.values():
        needed_ids.update(c)
    other_records = fetch_needed_records([cfg_paths.train_source2, cfg_paths.train_source3], needed_ids)
    log(f"  fetched {len(other_records)}/{len(needed_ids)} needed candidate records")

    log("Building labeled pair table (features + hard negatives from blocking) ...")
    pairs_df = build_pairs_dataframe(s1_records, other_records, candidates, gt, mcfg, seed)
    log(f"  {len(pairs_df)} pairs, positive rate = {pairs_df['label'].mean():.4f}")

    train_mask = pairs_df["s1_id"].isin(train_ids)
    val_mask = pairs_df["s1_id"].isin(val_ids)
    X_train = pairs_df.loc[train_mask, FEATURE_NAMES].values
    y_train = pairs_df.loc[train_mask, "label"].values
    X_val = pairs_df.loc[val_mask, FEATURE_NAMES].values
    y_val = pairs_df.loc[val_mask, "label"].values
    log(f"  train pairs={len(X_train)} (pos={y_train.sum()}), val pairs={len(X_val)} (pos={y_val.sum()})")

    results = {}
    for name, trainer in TRAINERS.items():
        log(f"Training {name} ...")
        try:
            model = trainer(X_train, y_train)
        except Exception as e:
            log(f"  {name} failed: {e}")
            continue
        val_proba = predict_proba(model, X_val)
        val_scored = pairs_df.loc[val_mask, ["s1_id", "cand_id"]].copy()
        val_scored["proba"] = val_proba

        search = search_best_threshold(val_scored, gt, list(val_ids), config.THRESHOLDS.candidates)
        best = search["best"]
        log(f"  {name}: best threshold={best['threshold']:.2f} "
            f"val macro entity F0.5={best['macro_entity_f05']:.4f} "
            f"avg_matches/matched_S1={best['avg_matches_per_matched_s1']:.2f}")
        results[name] = {"model": model, "search": search, "val_scored": val_scored}

    if not results:
        log("No model trained successfully — aborting.")
        return None

    best_name = max(results, key=lambda n: results[n]["search"]["best"]["macro_entity_f05"])
    best_entry = results[best_name]
    best_threshold = best_entry["search"]["best"]["threshold"]
    log(f"SELECTED MODEL: {best_name} (threshold={best_threshold}, "
        f"val macro entity F0.5={best_entry['search']['best']['macro_entity_f05']:.4f})")

    preds = decide_matches(best_entry["val_scored"], best_threshold)
    sfpr = singleton_false_positive_rate({s: gt[s] for s in val_ids if s in gt}, preds)
    log(f"  singleton false-positive rate on validation = {sfpr:.4f}")

    os.makedirs(config.PATHS.models_dir, exist_ok=True)
    import joblib
    joblib.dump(best_entry["model"], os.path.join(config.PATHS.models_dir, f"{best_name}_best.joblib"))

    return {
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "blocking_recall": b_recall,
        "avg_candidates_per_s1": avg_cand,
        "val_macro_entity_f05": best_entry["search"]["best"]["macro_entity_f05"],
        "singleton_fp_rate": sfpr,
        "all_model_results": {n: r["search"]["best"] for n, r in results.items()},
    }


def run_predict_test(model_path: str, threshold: float, index_row_cap: int = None):
    """Stage L/M: score the real test set and write the two required output
    files. Requires test_source1/2/3.tsv to exist under dataset/test/."""
    cfg_paths = config.PATHS
    bcfg = config.BLOCKING
    import joblib

    for p in [cfg_paths.test_source1, cfg_paths.test_source2, cfg_paths.test_source3]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — the test set has not been provided yet. "
                f"Place it under dataset/test/ before running predict-test."
            )

    log("Loading test Source1 ...")
    s1_df = read_source_full(cfg_paths.test_source1)
    all_test_s1_ids = s1_df["entity_id"].tolist()
    s1_records = {
        r.entity_id: (r.business_name, r.business_address or "", r.country)
        for r in s1_df.itertuples()
    }

    log("Pass 1/2: building blocking index over test Source2 + Source3 ...")
    idx = build_index_streaming([cfg_paths.test_source2, cfg_paths.test_source3], bcfg,
                                 row_cap=index_row_cap)

    log("Computing candidates for every test S1 ...")
    candidates: Dict[str, Set[str]] = {}
    for s1_id, (name, addr, country) in s1_records.items():
        candidates[s1_id] = idx.candidates_for(name, addr, country)

    log("Pass 2/2: fetching raw fields for candidate entities only ...")
    needed_ids = set()
    for c in candidates.values():
        needed_ids.update(c)
    other_records = fetch_needed_records([cfg_paths.test_source2, cfg_paths.test_source3], needed_ids)
    valid_s2 = {eid for eid in other_records if eid.startswith("S2-")}
    valid_s3 = {eid for eid in other_records if eid.startswith("S3-")}

    log(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)

    log("Scoring candidate pairs ...")
    from .features import compute_pair_features
    rows = []
    for s1_id, cand_ids in candidates.items():
        name1, addr1, country1 = s1_records[s1_id]
        for cid in cand_ids:
            if cid not in other_records:
                continue
            name2, addr2, country2 = other_records[cid]
            feats = compute_pair_features(name1, addr1, country1, name2, addr2, country2)
            rows.append({"s1_id": s1_id, "cand_id": cid, **feats})
    scored_df = pd.DataFrame(rows)
    if len(scored_df):
        X = scored_df[FEATURE_NAMES].values
        scored_df["proba"] = predict_proba(model, X)
    else:
        scored_df["proba"] = []

    matches = decide_matches(scored_df, threshold)

    log("Writing output files ...")
    write_matching_results(cfg_paths.matching_results, matches, all_test_s1_ids)
    write_candidate_pairs(cfg_paths.candidate_pairs, candidates, all_test_s1_ids)

    issues = self_check(matches, candidates, all_test_s1_ids, valid_s2, valid_s3)
    if issues:
        log(f"SELF-CHECK FOUND {len(issues)} ISSUE(S):")
        for i in issues[:20]:
            log(f"  - {i}")
    else:
        log("Self-check passed. Run utils/validate_submission.py to confirm before submitting.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train-eval")
    p1.add_argument("--max-s1", type=int, default=None)
    p1.add_argument("--val-fraction", type=float, default=0.2)
    p1.add_argument("--index-row-cap", type=int, default=None)

    p2 = sub.add_parser("predict-test")
    p2.add_argument("--model-path", type=str, required=True)
    p2.add_argument("--threshold", type=float, default=config.THRESHOLDS.default)
    p2.add_argument("--index-row-cap", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "train-eval":
        result = run_train_eval(max_s1=args.max_s1, val_fraction=args.val_fraction,
                                 index_row_cap=args.index_row_cap)
        print(json.dumps({k: v for k, v in result.items() if k != "all_model_results"}, indent=2))
    elif args.cmd == "predict-test":
        run_predict_test(args.model_path, args.threshold, index_row_cap=args.index_row_cap)


if __name__ == "__main__":
    main()
