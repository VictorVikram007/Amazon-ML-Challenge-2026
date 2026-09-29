"""Phase 0 data audit — chunked, memory-safe. Reads real files only."""
import pandas as pd
import numpy as np
from collections import Counter

TRAIN_DIR = "dataset/train"
CHUNK = 200_000

def audit_source_file(path, label):
    n_rows = 0
    missing = Counter()
    countries = Counter()
    name_lens = []
    addr_lens = []
    id_prefixes = Counter()
    dup_ids = Counter()
    name_len_sum = 0
    name_len_sumsq = 0
    addr_len_sum = 0
    addr_len_sumsq = 0
    n_name = 0
    n_addr = 0

    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False, na_values=[""]):
        n_rows += len(chunk)
        for col in ["entity_id", "business_name", "business_address", "country"]:
            missing[col] += chunk[col].isna().sum()
        countries.update(chunk["country"].dropna().value_counts().to_dict())
        id_prefixes.update(chunk["entity_id"].str.slice(0, 3).value_counts().to_dict())
        dup_ids.update(chunk["entity_id"].value_counts().to_dict())

        nl = chunk["business_name"].dropna().str.len()
        name_len_sum += nl.sum(); name_len_sumsq += (nl**2).sum(); n_name += len(nl)
        al = chunk["business_address"].dropna().str.len()
        addr_len_sum += al.sum(); addr_len_sumsq += (al**2).sum(); n_addr += len(al)

        # sample a few lengths for percentile approx (reservoir-lite: just keep first few chunks worth)
        if len(name_lens) < 200_000:
            name_lens.extend(nl.tolist()[:5000])
        if len(addr_lens) < 200_000:
            addr_lens.extend(al.tolist()[:5000])

    dup_count = sum(1 for k, v in dup_ids.items() if v > 1)

    print(f"\n===== {label} =====")
    print(f"rows: {n_rows}")
    print(f"missing per column: {dict(missing)}")
    print(f"id prefix counts: {dict(id_prefixes)}")
    print(f"duplicate entity_id count: {dup_count}")
    print(f"unique countries: {len(countries)} -> {countries.most_common(10)}")
    if n_name:
        mean_n = name_len_sum / n_name
        var_n = name_len_sumsq / n_name - mean_n**2
        print(f"business_name length: mean={mean_n:.2f} std={var_n**0.5:.2f} "
              f"min={min(name_lens) if name_lens else 'NA'} max={max(name_lens) if name_lens else 'NA'} "
              f"p50={np.percentile(name_lens,50):.1f} p95={np.percentile(name_lens,95):.1f}")
    if n_addr:
        mean_a = addr_len_sum / n_addr
        var_a = addr_len_sumsq / n_addr - mean_a**2
        print(f"business_address length: mean={mean_a:.2f} std={var_a**0.5:.2f} "
              f"min={min(addr_lens) if addr_lens else 'NA'} max={max(addr_lens) if addr_lens else 'NA'} "
              f"p50={np.percentile(addr_lens,50):.1f} p95={np.percentile(addr_lens,95):.1f}")
    return {
        "rows": n_rows, "missing": dict(missing), "countries": countries,
        "id_prefixes": dict(id_prefixes), "dup_count": dup_count,
    }

if __name__ == "__main__":
    s1 = audit_source_file(f"{TRAIN_DIR}/train_source1.tsv", "SOURCE 1 (train)")
    s2 = audit_source_file(f"{TRAIN_DIR}/train_source2.tsv", "SOURCE 2 (train)")
    s3 = audit_source_file(f"{TRAIN_DIR}/train_source3.tsv", "SOURCE 3 (train)")
