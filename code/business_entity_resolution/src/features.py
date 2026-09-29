"""
Pairwise feature engineering for a (Source1, candidate) pair.

Note on TF-IDF cosine: the spec's tip list mentions TF-IDF cosine similarity.
At competition scale (up to ~60 candidates x 2.2M S1 entities) fitting a
global TF-IDF matrix and doing full cosine search is exactly the kind of
Cartesian-style cost the brief tells us to avoid, and a *local* per-S1
TF-IDF fit adds real CPU cost for marginal gain over character n-gram
Jaccard, which is robust to the same typo noise (see audit: "Wanye"/"Wayne").
We therefore use n-gram Jaccard + rapidfuzz's C-implemented edit-distance
family (ratio / token_sort_ratio / token_set_ratio) as the continuous
similarity signals — cheap enough to compute per-pair at full scale, and
empirically this family captures the same "loose token/character overlap"
signal TF-IDF cosine would. This substitution is documented here and in the
methodology doc rather than silently claimed as literal TF-IDF.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

from .normalize import normalize_name, normalize_address, normalize_country, char_ngrams

FEATURE_NAMES: List[str] = [
    "name_exact", "name_token_jaccard", "name_char_ngram_jaccard",
    "name_edit_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_len_diff", "name_len_ratio", "name_common_prefix_ratio",
    "addr_exact", "addr_token_jaccard", "addr_char_ngram_jaccard",
    "addr_edit_ratio", "addr_numeric_overlap_count", "addr_numeric_jaccard",
    "addr_len_diff", "addr_len_ratio",
    "country_exact",
    "name_missing_either", "addr_missing_either",
    "sim_product", "sim_sum", "both_high", "strong_name_weak_addr", "weak_name_strong_addr",
    "exact_name_and_country", "exact_addr_and_country",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compute_pair_features(name1: str, addr1: str, country1: str,
                           name2: str, addr2: str, country2: str) -> Dict[str, float]:
    n1, n2 = normalize_name(name1), normalize_name(name2)
    a1, a2 = normalize_address(addr1), normalize_address(addr2)
    c1, c2 = normalize_country(country1), normalize_country(country2)

    name_tok1, name_tok2 = set(n1.tokens), set(n2.tokens)
    addr_tok1, addr_tok2 = set(a1.tokens), set(a2.tokens)
    name_grams1, name_grams2 = char_ngrams(n1.normalized, 3), char_ngrams(n2.normalized, 3)
    addr_grams1, addr_grams2 = char_ngrams(a1.normalized, 4), char_ngrams(a2.normalized, 4)
    num1, num2 = (a1.numeric_tokens or set()), (a2.numeric_tokens or set())

    name_edit = fuzz.ratio(n1.normalized, n2.normalized) / 100.0
    name_tsort = fuzz.token_sort_ratio(n1.normalized, n2.normalized) / 100.0
    name_tset = fuzz.token_set_ratio(n1.normalized, n2.normalized) / 100.0
    addr_edit = fuzz.ratio(a1.normalized, a2.normalized) / 100.0

    name_sim = max(name_edit, name_tsort, name_tset)
    addr_sim = addr_edit

    len1n, len2n = len(n1.normalized), len(n2.normalized)
    len1a, len2a = len(a1.normalized), len(a2.normalized)

    feats = {
        "name_exact": float(bool(n1.normalized) and n1.normalized == n2.normalized),
        "name_token_jaccard": _jaccard(name_tok1, name_tok2),
        "name_char_ngram_jaccard": _jaccard(name_grams1, name_grams2),
        "name_edit_ratio": name_edit,
        "name_token_sort_ratio": name_tsort,
        "name_token_set_ratio": name_tset,
        "name_len_diff": abs(len1n - len2n),
        "name_len_ratio": (min(len1n, len2n) / max(len1n, len2n)) if max(len1n, len2n) else 0.0,
        "name_common_prefix_ratio": (_common_prefix_len(n1.normalized, n2.normalized) /
                                      max(1, min(len1n, len2n))),
        "addr_exact": float(bool(a1.normalized) and a1.normalized == a2.normalized),
        "addr_token_jaccard": _jaccard(addr_tok1, addr_tok2),
        "addr_char_ngram_jaccard": _jaccard(addr_grams1, addr_grams2),
        "addr_edit_ratio": addr_edit,
        "addr_numeric_overlap_count": len(num1 & num2),
        "addr_numeric_jaccard": _jaccard(num1, num2),
        "addr_len_diff": abs(len1a - len2a),
        "addr_len_ratio": (min(len1a, len2a) / max(len1a, len2a)) if max(len1a, len2a) else 0.0,
        "country_exact": float(bool(c1) and c1 == c2),
        "name_missing_either": float(not n1.normalized or not n2.normalized),
        "addr_missing_either": float(not a1.normalized or not a2.normalized),
        "sim_product": name_sim * addr_sim,
        "sim_sum": name_sim + addr_sim,
        "both_high": float(name_sim > 0.8 and addr_sim > 0.8),
        "strong_name_weak_addr": float(name_sim > 0.85 and addr_sim < 0.5),
        "weak_name_strong_addr": float(name_sim < 0.5 and addr_sim > 0.85),
        "exact_name_and_country": float(n1.normalized == n2.normalized and c1 == c2 and bool(c1)),
        "exact_addr_and_country": float(a1.normalized == a2.normalized and c1 == c2 and bool(c1)),
    }
    return feats


def feature_vector(feats: Dict[str, float]) -> List[float]:
    return [feats[k] for k in FEATURE_NAMES]
