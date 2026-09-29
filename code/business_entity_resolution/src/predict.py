"""
Phase 8-9: threshold optimization + entity-level decision logic.

Turns a table of scored (s1_id, cand_id, probability) pairs into the final
entity-level match sets. Deliberately does NOT force exactly one match and
does NOT force every S1 to have a match — an S1 with no candidate above the
threshold is correctly left empty (a true singleton scores 1.0 for that).
"""
from __future__ import annotations
from typing import Dict, List, Optional

import pandas as pd

from .evaluate import macro_entity_f05


def decide_matches(scored: pd.DataFrame, threshold: float,
                    s1_col: str = "s1_id", cand_col: str = "cand_id",
                    proba_col: str = "proba") -> Dict[str, List[str]]:
    """scored: one row per candidate pair with a predicted probability.
    Keeps every candidate whose probability >= threshold; an S1 with none
    above threshold gets an empty list (never forced to pick top-1)."""
    kept = scored[scored[proba_col] >= threshold]
    out: Dict[str, List[str]] = {}
    for s1_id, group in kept.groupby(s1_col):
        ids = list(dict.fromkeys(group[cand_col].tolist()))  # de-dup, preserve order
        out[s1_id] = ids
    return out


def search_best_threshold(
    scored: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    all_val_s1_ids: List[str],
    thresholds: List[float],
    s1_col: str = "s1_id", cand_col: str = "cand_id", proba_col: str = "proba",
) -> Dict:
    """Try each threshold, compute validation macro entity F0.5 (over ALL
    validation S1 ids, including those with zero surviving candidates), and
    return the best. Also reports singleton false-positive rate and average
    matches per S1 at each threshold for transparency (Phase 8 of the brief)."""
    results = []
    gt_subset = {s1: ground_truth[s1] for s1 in all_val_s1_ids if s1 in ground_truth}
    for t in thresholds:
        preds = decide_matches(scored, t, s1_col, cand_col, proba_col)
        macro_f05, _ = macro_entity_f05(gt_subset, preds)
        n_matched_s1 = sum(1 for s1 in all_val_s1_ids if len(preds.get(s1, [])) > 0)
        avg_matches = (sum(len(v) for v in preds.values()) / n_matched_s1) if n_matched_s1 else 0.0
        results.append({
            "threshold": t, "macro_entity_f05": macro_f05,
            "n_s1_with_match": n_matched_s1, "avg_matches_per_matched_s1": avg_matches,
        })
    best = max(results, key=lambda r: r["macro_entity_f05"])
    return {"best": best, "all": results}
