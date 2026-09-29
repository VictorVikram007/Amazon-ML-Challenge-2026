"""
Phase 24: write the two required output files and self-check them against
every rule in the spec BEFORE the official utils/validate_submission.py runs
(catches obvious problems immediately, in-process, without a subprocess call).
"""
from __future__ import annotations
from typing import Dict, List, Set

from .io import write_id_list_tsv


def write_matching_results(path: str, matches: Dict[str, List[str]], all_test_s1_ids: List[str]) -> None:
    """Every test S1 must appear exactly once, even if matches has no entry for it."""
    rows = [(s1, matches.get(s1, [])) for s1 in all_test_s1_ids]
    write_id_list_tsv(path, rows, "source1_entity_id", "matched_entity_ids")


def write_candidate_pairs(path: str, candidates: Dict[str, Set[str]], all_test_s1_ids: List[str]) -> None:
    rows = [(s1, sorted(candidates.get(s1, []))) for s1 in all_test_s1_ids]
    write_id_list_tsv(path, rows, "source1_entity_id", "candidate_entity_ids")


def self_check(
    matches: Dict[str, List[str]],
    candidates: Dict[str, Set[str]],
    all_test_s1_ids: List[str],
    valid_s2_ids: Set[str],
    valid_s3_ids: Set[str],
) -> List[str]:
    """Re-implements every rule from the problem statement. Returns a list of
    human-readable issues; an empty list means it should pass the official
    validator too. This is a fast in-memory pre-check, not a replacement for
    running utils/validate_submission.py, which should still be run before
    every submission per the brief."""
    issues: List[str] = []
    s1_set = set(all_test_s1_ids)
    valid_ids = valid_s2_ids | valid_s3_ids

    missing_s1 = s1_set - set(matches.keys())
    # missing_s1 is fine as long as caller wrote empty rows for them — check via matches.get downstream.

    for s1 in all_test_s1_ids:
        ids = matches.get(s1, [])
        if len(ids) != len(set(ids)):
            issues.append(f"{s1}: duplicate IDs within matched_entity_ids")
        for i in ids:
            if i.startswith("S1-"):
                issues.append(f"{s1}: self-match to Source1 id {i}")
            elif i not in valid_ids:
                issues.append(f"{s1}: matched id {i} not found in test Source2/Source3")
        cand = candidates.get(s1, set())
        not_in_cand = [i for i in ids if i not in cand]
        if not_in_cand:
            issues.append(f"{s1}: matched id(s) {not_in_cand} not present in candidate_pairs (pipeline bug)")

    n_rows = len({s1 for s1 in all_test_s1_ids})
    if n_rows != len(all_test_s1_ids):
        issues.append("duplicate source1_entity_id rows in output")

    return issues
