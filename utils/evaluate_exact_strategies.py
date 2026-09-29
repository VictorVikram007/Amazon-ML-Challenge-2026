#!/usr/bin/env python3
import csv
import hashlib
import sqlite3
from pathlib import Path
from src.evaluate import macro_entity_f05
from src.io import iter_ground_truth
from src.normalize import normalize_address, normalize_country, normalize_name

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"
DB = ROOT / "output" / "train_exact.sqlite3"

def h(country, kind, value):
    raw = f"{country}\x1f{kind}\x1f{value}".encode()
    n = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return n - (1 << 64) if n >= (1 << 63) else n

def add(conn, path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f, delimiter="\t"):
            c = normalize_country(r["country"])
            n = normalize_name(r["business_name"]).normalized
            a = normalize_address(r["business_address"]).normalized
            if n: rows.append((h(c, "n", n), r["entity_id"]))
            if a: rows.append((h(c, "a", a), r["entity_id"]))
            if len(rows) >= 10000:
                conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?)", rows); rows.clear()
        if rows: conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?)", rows)
    conn.commit()

def ids(conn, c, kind, value):
    if not value: return set()
    return {r[0] for r in conn.execute("SELECT entity_id FROM exact WHERE key_hash=?", (h(c, kind, value),))}

def main():
    DB.unlink(missing_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE exact (key_hash INTEGER, entity_id TEXT, PRIMARY KEY(key_hash, entity_id)) WITHOUT ROWID")
    add(conn, TRAIN / "train_source2.tsv"); add(conn, TRAIN / "train_source3.tsv")
    gt = {}
    for ch in iter_ground_truth(str(TRAIN / "train_ground_truth.tsv")):
        for sid, value in zip(ch.source1_entity_id, ch.matched_entity_ids): gt[sid] = value.split(",") if value else []
    preds = {"address": {}, "name": {}, "union": {}, "unique": {}, "intersection": {}}
    with (TRAIN / "train_source1.tsv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            c = normalize_country(r["country"]); n = ids(conn, c, "n", normalize_name(r["business_name"]).normalized); a = ids(conn, c, "a", normalize_address(r["business_address"]).normalized)
            preds["address"][r["entity_id"]] = sorted(a)
            preds["name"][r["entity_id"]] = sorted(n)
            preds["union"][r["entity_id"]] = sorted(n | a)
            preds["unique"][r["entity_id"]] = sorted(a if a else (n if len(n) == 1 else set()))
            preds["intersection"][r["entity_id"]] = sorted(n & a if n and a else (a if a else set()))
    for name, pred in preds.items():
        score, _ = macro_entity_f05(gt, pred); print(name, f"{score:.6f}")
    conn.close(); DB.unlink(missing_ok=True)

if __name__ == "__main__": main()
