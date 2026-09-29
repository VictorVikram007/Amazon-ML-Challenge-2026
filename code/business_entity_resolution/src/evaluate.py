"""
The PRIMARY local metric: entity-level macro F0.5, computed exactly the way
the competition scores it (per S1 entity, then macro-averaged over ALL S1
entities including singletons — a correctly-predicted empty set scores 1.0).

Also provides pair-level precision/recall/F0.5 and blocking-recall
diagnostics, used as secondary signals during development (per Phase 2/4/8
of the brief) — never as the model-selection criterion.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """Per-entity F0.5. A correctly-predicted empty set (true singleton,
    predicted empty) scores 1.0; predicting anything for a true singleton
    scores 0.0; missing a true match set (true non-empty, predicted empty)
    scores 0.0 (precision undefined -> treated as 0)."""
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids:  # true has matches, we predicted none: recall=0
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids) if true_ids else 0.0
    return f_beta(precision, recall, beta=0.5)


def macro_entity_f05(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
) -> Tuple[float, Dict[str, float]]:
    """Macro-average entity F0.5 over every S1 id present in ground_truth.
    Any S1 missing from `predictions` is treated as an empty prediction."""
    scores: Dict[str, float] = {}
    for s1, true_list in ground_truth.items():
        pred_list = predictions.get(s1, [])
        scores[s1] = entity_f05(set(true_list), set(pred_list))
    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return macro, scores


def singleton_false_positive_rate(ground_truth: Dict[str, List[str]],
                                   predictions: Dict[str, List[str]]) -> float:
    """Of all true singletons (zero true matches), fraction we incorrectly gave a match."""
    singles = [s1 for s1, m in ground_truth.items() if len(m) == 0]
    if not singles:
        return 0.0
    fp = sum(1 for s1 in singles if len(predictions.get(s1, [])) > 0)
    return fp / len(singles)


def pair_level_prf(true_pairs: Set[Tuple[str, str]], pred_pairs: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    if not pred_pairs and not true_pairs:
        return 1.0, 1.0, 1.0
    tp = len(true_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(true_pairs) if true_pairs else 0.0
    return precision, recall, f_beta(precision, recall, 0.5)


def blocking_recall(ground_truth: Dict[str, List[str]],
                     candidates: Dict[str, Set[str]]) -> Tuple[float, float, float]:
    """Fraction of true (S1, matched_id) pairs that survive blocking, plus
    average and max candidate-set size — the ceiling on achievable recall."""
    total_true = 0
    recovered = 0
    sizes = []
    for s1, true_list in ground_truth.items():
        cands = candidates.get(s1, set())
        sizes.append(len(cands))
        for m in true_list:
            total_true += 1
            if m in cands:
                recovered += 1
    recall = recovered / total_true if total_true else 1.0
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0
    max_size = max(sizes) if sizes else 0
    return recall, avg_size, max_size
