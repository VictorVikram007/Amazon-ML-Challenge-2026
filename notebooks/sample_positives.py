"""Load a manageable slice: first N ground-truth rows with multi-match,
then pull the actual S1/S2/S3 records for those IDs to see real noisy variations."""
import pandas as pd

N_GT = 3000  # small sample of S1 entities to inspect closely

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str,
                  keep_default_na=False, na_values=[""], nrows=N_GT)

needed_s1 = set(gt["source1_entity_id"])
needed_s2, needed_s3 = set(), set()
for m in gt["matched_entity_ids"].fillna(""):
    if not m:
        continue
    for i in m.split(","):
        if i.startswith("S2-"):
            needed_s2.add(i)
        elif i.startswith("S3-"):
            needed_s3.add(i)

def load_needed(path, needed_ids, chunksize=300_000):
    rows = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize,
                              keep_default_na=False, na_values=[""]):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        if len(hit):
            rows.append(hit)
    return pd.concat(rows) if rows else pd.DataFrame(columns=["entity_id","business_name","business_address","country"])

s1 = load_needed("dataset/train/train_source1.tsv", needed_s1)
s2 = load_needed("dataset/train/train_source2.tsv", needed_s2)
s3 = load_needed("dataset/train/train_source3.tsv", needed_s3)

print(f"Found {len(s1)}/{len(needed_s1)} S1, {len(s2)}/{len(needed_s2)} S2, {len(s3)}/{len(needed_s3)} S3 needed records")

s1_idx = s1.set_index("entity_id")
s2_idx = s2.set_index("entity_id")
s3_idx = s3.set_index("entity_id")

print("\n===== 20 REAL POSITIVE MATCH GROUPS =====")
shown = 0
for _, row in gt.iterrows():
    if shown >= 20:
        break
    s1id = row["source1_entity_id"]
    m = row["matched_entity_ids"]
    if not m or s1id not in s1_idx.index:
        continue
    r1 = s1_idx.loc[s1id]
    print(f"\n[S1] {s1id} | NAME: {r1['business_name']!r} | ADDR: {r1['business_address']!r} | {r1['country']}")
    for mid in m.split(","):
        idx = s2_idx if mid.startswith("S2-") else s3_idx
        if mid in idx.index:
            r = idx.loc[mid]
            print(f"   [{mid}] NAME: {r['business_name']!r} | ADDR: {r['business_address']!r} | {r['country']}")
    shown += 1

# leakage / referential integrity check on this sample
missing_s2 = needed_s2 - set(s2["entity_id"])
missing_s3 = needed_s3 - set(s3["entity_id"])
print(f"\nMissing S2 ids not found in source2 (sampled slice): {len(missing_s2)}")
print(f"Missing S3 ids not found in source3 (sampled slice): {len(missing_s3)}")
