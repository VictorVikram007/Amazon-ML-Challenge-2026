#!/usr/bin/env python3
import csv
import hashlib
import sqlite3
from pathlib import Path
from src.evaluate import entity_f05
from src.io import iter_ground_truth
from src.normalize import normalize_address, normalize_country, normalize_name

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"
DB = ROOT / "output" / "train_policy.sqlite3"

def key_hash(c, kind, value):
    raw = f"{c}\x1f{kind}\x1f{value}".encode()
    n = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return n - (1 << 64) if n >= (1 << 63) else n

def add_source(conn, path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f, delimiter="\t"):
            c = normalize_country(r["country"])
            n = normalize_name(r["business_name"]).normalized
            a = normalize_address(r["business_address"]).normalized
            if n: rows.append((key_hash(c, "n", n), r["entity_id"]))
            if a: rows.append((key_hash(c, "a", a), r["entity_id"]))
            if len(rows) >= 10000:
                conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?)", rows); rows.clear()
        if rows: conn.executemany("INSERT OR IGNORE INTO exact VALUES (?, ?)", rows)
    conn.commit()

def fetch(conn, c, kind, value):
    if not value: return set()
    return {x[0] for x in conn.execute("SELECT entity_id FROM exact WHERE key_hash=?", (key_hash(c, kind, value),))}

def main():
    DB.unlink(missing_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=OFF"); conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE exact (key_hash INTEGER, entity_id TEXT, PRIMARY KEY(key_hash, entity_id)) WITHOUT ROWID")
    add_source(conn, TRAIN / "train_source2.tsv"); add_source(conn, TRAIN / "train_source3.tsv")
    gt = {}
    for ch in iter_ground_truth(str(TRAIN / "train_ground_truth.tsv")):
        for sid, value in zip(ch.source1_entity_id, ch.matched_entity_ids): gt[sid] = set(value.split(",")) if value else set()
    totals = {"address": 0.0, "unique_name": 0.0, "address_or_unique_name": 0.0}
    counts = {k: 0 for k in totals}; n = 0
    with (TRAIN / "train_source1.tsv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sid = r["entity_id"]; c = normalize_country(r["country"])
            names = fetch(conn, c, "n", normalize_name(r["business_name"]).normalized)
            addrs = fetch(conn, c, "a", normalize_address(r["business_address"]).normalized)
            policies = {"address": addrs, "unique_name": names if len(names) == 1 else set(), "address_or_unique_name": addrs if addrs else (names if len(names) == 1 else set())}
            for k, pred in policies.items():
                totals[k] += entity_f05(gt.get(sid, set()), pred)
                counts[k] += bool(pred)
            n += 1
            if n % 100000 == 0: print(f"processed {n}", flush=True)
    print(f"entities={n}")
    for k in totals: print(f"{k}: score={totals[k]/n:.6f} nonempty={counts[k]}")
    conn.close(); DB.unlink(missing_ok=True)

if __name__ == "__main__": main()
