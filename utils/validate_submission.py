#!/usr/bin/env python3
"""
Stdlib-only validator for matching_results.tsv and candidate_pairs.tsv,
implementing every rule from the competition problem statement:

  - every test Source 1 entity appears exactly once in matching_results.tsv
    (and, checked the same way, in candidate_pairs.tsv)
  - no duplicate source1_entity_id rows
  - no duplicate IDs within a single ID list
  - matched/candidate IDs must be S2-/S3- IDs that exist in the test set
  - no S1 self-matches
  - matching_results' matched IDs must be a subset of that S1's candidate_pairs entry

Usage:
  python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test

Prints PASS (exit 0) or a numbered list of issues (exit 1). Reads only the
output files and the test source files; does not compute the score.
"""
import argparse
import csv
import sys
from collections import Counter


def load_tsv_ids(path):
    """Read a two-column id-list TSV -> dict[id] = [list of ids], and report
    duplicate keys plus raw row count for the duplicate-row check."""
    rows = {}
    dup_keys = []
    seen_order = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for line in reader:
            if not line:
                continue
            key = line[0]
            val = line[1] if len(line) > 1 else ""
            if key in rows:
                dup_keys.append(key)
            rows[key] = val.split(",") if val else []
            seen_order.append(key)
    return rows, dup_keys, seen_order, header


def load_test_entity_ids(test_dir):
    import os
    s1_ids, s2_ids, s3_ids = set(), set(), set()
    for fname, target in [("test_source1.tsv", s1_ids), ("test_source2.tsv", s2_ids), ("test_source3.tsv", s3_ids)]:
        path = os.path.join(test_dir, fname)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for line in reader:
                if line:
                    target.add(line[0])
    return s1_ids, s2_ids, s3_ids


def validate(matching_path, candidate_path, test_dir):
    issues = []
    s1_ids, s2_ids, s3_ids = load_test_entity_ids(test_dir)
    valid_ids = s2_ids | s3_ids

    match_rows, match_dups, match_order, match_header = load_tsv_ids(matching_path)
    cand_rows, cand_dups, cand_order, cand_header = load_tsv_ids(candidate_path)

    if match_header != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv header is {match_header}, expected "
                       f"['source1_entity_id', 'matched_entity_ids']")
    if cand_header != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv header is {cand_header}, expected "
                       f"['source1_entity_id', 'candidate_entity_ids']")

    if match_dups:
        issues.append(f"matching_results.tsv: {len(match_dups)} duplicate source1_entity_id rows, "
                       f"e.g. {match_dups[:5]}")
    if cand_dups:
        issues.append(f"candidate_pairs.tsv: {len(cand_dups)} duplicate source1_entity_id rows, "
                       f"e.g. {cand_dups[:5]}")

    missing_in_match = s1_ids - set(match_rows.keys())
    if missing_in_match:
        issues.append(f"matching_results.tsv: {len(missing_in_match)} test S1 entities missing, "
                       f"e.g. {list(missing_in_match)[:5]}")
    extra_in_match = set(match_rows.keys()) - s1_ids
    if extra_in_match:
        issues.append(f"matching_results.tsv: {len(extra_in_match)} rows for S1 ids not in test set, "
                       f"e.g. {list(extra_in_match)[:5]}")

    missing_in_cand = s1_ids - set(cand_rows.keys())
    if missing_in_cand:
        issues.append(f"candidate_pairs.tsv: {len(missing_in_cand)} test S1 entities missing, "
                       f"e.g. {list(missing_in_cand)[:5]}")

    n_bad_dup_ids = 0
    n_self_match = 0
    n_invalid_id = 0
    n_examples = []
    for s1, ids in match_rows.items():
        if len(ids) != len(set(ids)):
            n_bad_dup_ids += 1
        for i in ids:
            if i.startswith("S1-"):
                n_self_match += 1
                if len(n_examples) < 5:
                    n_examples.append(f"{s1} -> self-match {i}")
            elif i not in valid_ids:
                n_invalid_id += 1
                if len(n_examples) < 5:
                    n_examples.append(f"{s1} -> unknown id {i}")
    if n_bad_dup_ids:
        issues.append(f"matching_results.tsv: {n_bad_dup_ids} rows contain duplicate IDs within their list")
    if n_self_match:
        issues.append(f"matching_results.tsv: {n_self_match} self-matches to Source1. Examples: {n_examples}")
    if n_invalid_id:
        issues.append(f"matching_results.tsv: {n_invalid_id} matched IDs not present in test Source2/3")

    # subset check: every matched id must appear in that S1's candidate list
    subset_violations = 0
    subset_examples = []
    for s1, ids in match_rows.items():
        cand_set = set(cand_rows.get(s1, []))
        bad = [i for i in ids if i not in cand_set]
        if bad:
            subset_violations += 1
            if len(subset_examples) < 5:
                subset_examples.append((s1, bad))
    if subset_violations:
        issues.append(f"WARNING: {subset_violations} S1 entities have matched id(s) not present in "
                       f"candidate_pairs.tsv (pipeline inconsistency). Examples: {subset_examples}")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL — {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
