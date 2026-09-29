"""Ground-truth audit: match-count distribution, S2 vs S3 split, sample positives."""
import pandas as pd
from collections import Counter

CHUNK = 200_000
path = "dataset/train/train_ground_truth.tsv"

n_rows = 0
zero = one = multi = 0
total_matches = 0
s2_count = 0
s3_count = 0
match_count_hist = Counter()
dup_s1 = Counter()

for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False, na_values=[""]):
    n_rows += len(chunk)
    dup_s1.update(chunk["source1_entity_id"].value_counts().to_dict())
    mids = chunk["matched_entity_ids"].fillna("")
    for m in mids:
        if m == "":
            zero += 1
            match_count_hist[0] += 1
            continue
        ids = m.split(",")
        k = len(ids)
        match_count_hist[k] += 1
        if k == 1:
            one += 1
        else:
            multi += 1
        total_matches += k
        for i in ids:
            if i.startswith("S2-"):
                s2_count += 1
            elif i.startswith("S3-"):
                s3_count += 1

dup_count = sum(1 for k, v in dup_s1.items() if v > 1)

print(f"rows: {n_rows}")
print(f"duplicate source1_entity_id rows: {dup_count}")
print(f"zero-match S1: {zero} ({100*zero/n_rows:.2f}%)")
print(f"one-match S1: {one} ({100*one/n_rows:.2f}%)")
print(f"multi-match S1: {multi} ({100*multi/n_rows:.2f}%)")
print(f"avg matches per S1 (over all S1, incl. zero): {total_matches/n_rows:.3f}")
print(f"avg matches per S1 (over matched only): {total_matches/(one+multi):.3f}")
print(f"total matched-id references: {total_matches}  S2-refs: {s2_count} ({100*s2_count/total_matches:.1f}%)  S3-refs: {s3_count} ({100*s3_count/total_matches:.1f}%)")
print("match-count histogram (top 15):", sorted(match_count_hist.items())[:15])
