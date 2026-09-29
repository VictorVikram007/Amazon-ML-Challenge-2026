import polars as pl, sys, os, gc
sys.path.insert(0, 'work'); from norm import load, tokens
split = sys.argv[1]; os.makedirs(f'work/{split}', exist_ok=True)
ONLY = sys.argv[2] if len(sys.argv) > 2 else 'all'   # build tokens for one country only
DROP = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # % of train S1 to hide (their records become decoys, like test)
for s in ['1', 'o']:
    files = [f'dataset/{split}/{split}_source1.tsv'] if s == '1' else \
            [f'dataset/{split}/{split}_source2.tsv', f'dataset/{split}/{split}_source3.tsv']
    df = pl.concat([load(f).select('entity_id', 'country', 'nm', 'ad', 'sk', 'core', 'codes', 'business_name', 'business_address') for f in files]).with_row_index('rid')
    if s == '1' and DROP:
        df = df.drop('rid').filter(pl.col('entity_id').hash(42) % 100 >= DROP).with_row_index('rid')
    df.write_parquet(f'work/{split}/s{s}.parquet')
    df = df.select('rid', 'country', 'nm', 'ad', 'sk', 'codes', 'core'); gc.collect()
    for c in df['country'].unique().to_list():
        if ONLY != 'all' and c != ONLY: continue
        part = df.filter(pl.col('country') == c)
        path = f'work/{split}/tok_{s}_{c}'; os.makedirs(path, exist_ok=True)
        for j, i in enumerate(range(0, part.height, 300_000)):
            tokens(part.slice(i, 300_000)).sort('rid').write_parquet(f'{path}/{j:04d}.parquet')
        part.filter(pl.col('core').str.len_chars() >= 3).select('rid', tok=pl.col('core').hash(9)).write_parquet(f'work/{split}/xtok_{s}_{c}.parquet')
        print(split, s, c, flush=True); del part; gc.collect()
    del df; gc.collect()
