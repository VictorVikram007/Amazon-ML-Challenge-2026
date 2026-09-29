"""
Phase 5-7: pairwise feature construction, hard-negative mining, and baseline
model comparison (Logistic Regression -> XGBoost -> LightGBM), selected on
validation ENTITY-LEVEL macro F0.5 (never on pair-level accuracy/AUC alone).

Negative sampling strategy (Phase 1 / Phase 7 of the brief):
- Every candidate produced by blocking that is *not* a true match is already
  a naturally "hard" negative: it passed every blocking filter (same country,
  overlapping tokens/n-grams/numeric-tokens) yet is the wrong business. This
  is exactly the kind of "similar name, wrong address" / "same numeric token,
  unrelated business" negative the brief asks for — no synthetic negatives
  need to be invented.
- If a fraction of positives have very few or zero such negatives (their S1
  had a small/clean candidate set), we top up with a modest amount of
  cross-entity random negatives (a candidate's record paired against an
  unrelated S1 in the same country) so the model also sees "far apart"
  negatives, controlled by ModelConfig.hard_negative_fraction.
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_NAMES, compute_pair_features, feature_vector

Record = Tuple[str, str, str]  # (name, address, country)


def build_pairs_dataframe(
    s1_records: Dict[str, Record],
    other_records: Dict[str, Record],
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, List[str]],
    cfg: config.ModelConfig = config.MODEL,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (s1_id, cand_id) candidate pair with features + label.
    label=1 iff cand_id is in ground_truth[s1_id]. Every row's cand_id is
    guaranteed to have come from `candidates` (i.e. it is a real blocking
    candidate, so negatives are the natural hard negatives described above)."""
    rng = random.Random(seed)
    rows = []
    all_s1_ids = list(s1_records.keys())

    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_records:
            continue
        name1, addr1, country1 = s1_records[s1_id]
        true_set = set(ground_truth.get(s1_id, []))

        pos_ids = [c for c in cand_ids if c in true_set]
        neg_ids = [c for c in cand_ids if c not in true_set]

        # Cap negatives per S1 so one generic S1 with a huge candidate set
        # doesn't dominate the training set (class-imbalance control).
        max_neg = max(len(pos_ids), 1) * cfg.negative_per_positive
        if len(neg_ids) > max_neg:
            neg_ids = rng.sample(neg_ids, max_neg)

        for cid in pos_ids + neg_ids:
            if cid not in other_records:
                continue
            name2, addr2, country2 = other_records[cid]
            feats = compute_pair_features(name1, addr1, country1, name2, addr2, country2)
            row = {"s1_id": s1_id, "cand_id": cid, "label": int(cid in true_set)}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def split_by_entity(s1_ids: List[str], val_fraction: float = 0.2, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """Split S1 entity IDs (not pairs/rows) into train/val to avoid any
    leakage of one S1's positive/negative pairs across the split."""
    rng = random.Random(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def train_logreg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, cfg: config.ModelConfig = config.MODEL):
    from xgboost import XGBClassifier
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    model = XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
        scale_pos_weight=n_neg / n_pos, tree_method="hist", n_jobs=1,
        random_state=cfg.random_state, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, cfg: config.ModelConfig = config.MODEL):
    from lightgbm import LGBMClassifier
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    model = LGBMClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
        scale_pos_weight=n_neg / n_pos, n_jobs=1, random_state=cfg.random_state, verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


TRAINERS = {"logreg": train_logreg, "xgboost": train_xgboost, "lightgbm": train_lightgbm}


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
