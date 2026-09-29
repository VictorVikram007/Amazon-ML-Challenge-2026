#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path
from src.evaluate import macro_entity_f05
from src.io import iter_ground_truth
from src.normalize import normalize_address, normalize_country, normalize_name

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"
SOURCE_LIMIT = 300_000
S1_LIMIT = 300_000

def load_source(path):
    by_name = defaultdict(set)
    by_address = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if i >= SOURCE_LIMIT: break
            c = normalize_country(row["country"])
            n = normalize_name(row["business_name"]).normalized
            a = normalize_address(row["business_address"]).normalized
            if n: by_name[(c, n)].add(row["entity_id"])
            if a: by_address[(c, a)].add(row["entity_id"])
    return by_name, by_address

def main():
    name2, addr2 = load_source(TRAIN / "train_source2.tsv")
    name3, addr3 = load_source(TRAIN / "train_source3.tsv")
    names = defaultdict(set); addrs = defaultdict(set)
    for k, v in name2.items(): names[k].update(v)
    for k, v in name3.items(): names[k].update(v)
    for k, v in addr2.items(): addrs[k].update(v)
    for k, v in addr3.items(): addrs[k].update(v)
    gt = {}
    for chunk in iter_ground_truth(str(TRAIN / "train_ground_truth.tsv")):
        for sid, ids in zip(chunk.source1_entity_id, chunk.matched_entity_ids): gt[sid] = ids.split(",") if ids else []
    preds = {x: {} for x in ("address", "unique_name", "address_or_unique_name", "intersection")}
    with (TRAIN / "train_source1.tsv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if i >= S1_LIMIT: break
            sid = row["entity_id"]; c = normalize_country(row["country"])
            n = names.get((c, normalize_name(row["business_name"]).normalized), set())
            a = addrs.get((c, normalize_address(row["business_address"]).normalized), set())
            preds["address"][sid] = sorted(a)
            preds["unique_name"][sid] = sorted(n) if len(n) == 1 else []
            preds["address_or_unique_name"][sid] = sorted(a if a else (n if len(n) == 1 else set()))
            preds["intersection"][sid] = sorted(a & n) if a and n else []
    gt_slice = {k: gt[k] for k in preds["address"]}
    for policy, pred in preds.items():
        score, _ = macro_entity_f05(gt_slice, pred)
        nonempty = sum(bool(v) for v in pred.values())
        print(f"{policy}: score={score:.6f} nonempty={nonempty}")

if __name__ == "__main__": main()
