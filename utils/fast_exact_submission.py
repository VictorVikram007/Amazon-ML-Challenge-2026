#!/usr/bin/env python3
"""Create a fast bounded exact-normalized test submission."""
import csv
import hashlib
import sqlite3
from pathlib import Path

from src.normalize import normalize_address, normalize_country, normalize_name


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
DB_PATH = OUTPUT_DIR / "exact_lookup.sqlite3"


def key_hash(country, kind, value):
    raw = f"{country}\x1f{kind}\x1f{value}".encode("utf-8")
    number = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return number - (1 << 64) if number >= (1 << 63) else number


def add_source(conn, path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            country = normalize_country(row["country"])
            name = normalize_name(row["business_name"]).normalized
            address = normalize_address(row["business_address"]).normalized
            if name:
                rows.append((key_hash(country, "n", name), row["entity_id"], address))
            if address:
                rows.append((key_hash(country, "a", address), row["entity_id"], address))
            if len(rows) >= 10000:
                conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?, ?)", rows)
                rows.clear()
        if rows:
            conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?, ?)", rows)


def find_matches(conn, country, name, address):
    found = set()
    if address:
        found.update(row[0] for row in conn.execute(
            "SELECT entity_id FROM exact WHERE key_hash=?",
            (key_hash(country, "a", address),)
        ))
    if name:
        source_tokens = set(address.split())
        for entity_id, candidate_address in conn.execute(
            "SELECT entity_id, address FROM exact WHERE key_hash=?",
            (key_hash(country, "n", name),)
        ):
            candidate_tokens = set(candidate_address.split())
            if not candidate_tokens or not source_tokens or source_tokens & candidate_tokens:
                found.add(entity_id)
    return sorted(found)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("CREATE TABLE exact (key_hash INTEGER NOT NULL, entity_id TEXT NOT NULL, address TEXT NOT NULL, PRIMARY KEY (key_hash, entity_id)) WITHOUT ROWID")
    add_source(conn, TEST_DIR / "test_source2.tsv")
    add_source(conn, TEST_DIR / "test_source3.tsv")

    with (TEST_DIR / "test_source1.tsv").open(newline="", encoding="utf-8") as source, \
         (OUTPUT_DIR / "matching_results.tsv").open("w", newline="", encoding="utf-8") as matching, \
         (OUTPUT_DIR / "candidate_pairs.tsv").open("w", newline="", encoding="utf-8") as candidates:
        reader = csv.DictReader(source, delimiter="\t")
        match_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidates, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in reader:
            country = normalize_country(row["country"])
            name = normalize_name(row["business_name"]).normalized
            address = normalize_address(row["business_address"]).normalized
            ids = find_matches(conn, country, name, address)
            joined = ",".join(ids)
            match_writer.writerow([row["entity_id"], joined])
            candidate_writer.writerow([row["entity_id"], joined])
    conn.close()
    DB_PATH.unlink(missing_ok=True)
if __name__ == "__main__":
    main()