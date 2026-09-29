"""
Memory-safe I/O helpers.

Every source file is read with sep="\t" and dtype=str (never inferred),
because entity IDs, PIN codes, etc. must never be silently coerced to
numbers. Large files are streamed in chunks wherever the caller does not
need the whole file resident in memory at once.
"""
from __future__ import annotations
import csv
import os
from typing import Dict, Iterator, Iterable, List, Optional, Tuple

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def iter_source_chunks(path: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield source TSV chunks without pandas parser-sized allocations."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected source header in {path}: {header}")

        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(SOURCE_COLUMNS):
                raise ValueError(f"Malformed row in {path}: expected 4 columns, got {len(row)}")
            rows.append(row)
            if len(rows) >= chunksize:
                yield pd.DataFrame(rows, columns=SOURCE_COLUMNS)
                rows = []
        if rows:
            yield pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def read_source_full(path: str) -> pd.DataFrame:
    """Read an entire source file into memory. Only use for files known to be
    small enough (e.g. test_source1 / train_source1 on a machine with enough RAM).
    For source2/source3 at competition scale, prefer iter_source_chunks."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def iter_ground_truth(path: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, chunksize=chunksize,
        keep_default_na=False, na_values=[""],
    ):
        chunk["matched_entity_ids"] = chunk["matched_entity_ids"].fillna("")
        yield chunk


def load_ground_truth_map(path: str) -> Dict[str, List[str]]:
    """Load the full ground-truth mapping S1 -> [matched ids]. train_ground_truth.tsv
    is ~2.2M rows / ~127MB, which fits as a plain dict of lists on a normal machine."""
    gt: Dict[str, List[str]] = {}
    for chunk in iter_ground_truth(path):
        for s1, matched in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            gt[s1] = matched.split(",") if matched else []
    return gt


def write_id_list_tsv(path: str, rows: Iterable[Tuple[str, List[str]]], id_col: str, list_col: str) -> None:
    """Write a TSV with columns [id_col, list_col], list_col rendered as a
    comma-joined string with no quoting (matches the spec's example format exactly)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow([id_col, list_col])
        for entity_id, ids in rows:
            # de-duplicate while preserving order, never write "nan"/None
            seen = set()
            clean = []
            for i in ids:
                if i and i not in seen:
                    seen.add(i)
                    clean.append(i)
            writer.writerow([entity_id, ",".join(clean)])


def load_id_list_tsv(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    id_col, list_col = df.columns[0], df.columns[1]
    for k, v in zip(df[id_col], df[list_col].fillna("")):
        out[k] = v.split(",") if v else []
    return out
