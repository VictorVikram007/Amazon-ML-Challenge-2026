"""Group-level features: compare each (S1, record) pair against records confidently assigned to the same S1."""
import polars as pl, numpy as np, sys, gc, time
from rapidfuzz import process, fuzz
split, predf = sys.argv[1], sys.argv[2]
F = pl.read_parquet(f'work/{split}/{predf}')
F = F.with_columns(prk=pl.col('p').rank('ordinal', descending=True).over('ro'))
p2 = F.filter(pl.col('prk') == 2).select('ro', p_2nd='p')
B = F.filter(pl.col('prk') == 1, pl.col('p') > 0.02).drop('prk').join(p2, on='ro', how='left').with_columns(pl.col('p_2nd').fill_null(0))
del F; gc.collect()
B = B.with_columns(g_n=(pl.col('p') > 0.5).sum().over('r1').cast(pl.UInt16), g_all=pl.len().over('r1').cast(pl.UInt16),
                   g_sum=pl.col('p').sum().over('r1'), g_rank=pl.col('p').rank('ordinal', descending=True).over('r1').cast(pl.UInt16),
                   g_min_conf=pl.col('p').filter(pl.col('p') > 0.5).min().over('r1'))
so = pl.read_parquet(f'work/{split}/so.parquet', columns=['rid', 'nm', 'ad']).rename({'rid': 'ro'})
M = B.filter(pl.col('p') > 0.5).select('r1', rm='ro')
def cp(a, b, sc): return process.cpdist(a, b, scorer=sc, workers=-1).astype(np.float32)
outs = []; lo, hi = B['ro'].min(), B['ro'].max(); step = (hi - lo) // 8 + 1
for q in range(8):
    t = time.time()
    x = B.filter(pl.col('ro').is_between(lo + q*step, lo + (q+1)*step - 1)).select('r1', 'ro').join(M, on='r1').filter(pl.col('ro') != pl.col('rm'))
    x = x.join(so, on='ro').join(so.rename({'ro': 'rm', 'nm': 'nm_m', 'ad': 'ad_m'}), on='rm')
    a, am, n, nm = [x[c].to_list() for c in ['ad', 'ad_m', 'nm', 'nm_m']]
    x = x.select('r1', 'ro', m_a_tset=cp(a, am, fuzz.token_set_ratio), m_a_ratio=cp(a, am, fuzz.ratio),
                 m_n_tset=cp(n, nm, fuzz.token_set_ratio), m_n_ratio=cp(n, nm, fuzz.ratio),
                 m_a_eq=(x['ad'] == x['ad_m']) & (x['ad'].str.len_chars() > 0))
    outs.append(x.group_by('r1', 'ro').agg(pl.col('m_a_tset', 'm_a_ratio', 'm_n_tset', 'm_n_ratio').max(),
                 m_a_eq=pl.col('m_a_eq').sum().cast(pl.UInt16), m_a_mean=pl.col('m_a_tset').mean()))
    del x, a, am, n, nm; gc.collect(); print(q, round(time.time() - t), 's', flush=True)
G = pl.concat(outs)
B = B.join(G, on=['r1', 'ro'], how='left')
# add stage-1 pair features back
F1 = pl.scan_parquet(f'work/{split}/feat/*.parquet').join(B.select('r1', 'ro').lazy(), on=['r1', 'ro'], how='semi').collect(engine='streaming')
B = B.join(F1, on=['r1', 'ro'], how='left'); B.write_parquet(f'work/{split}/s2.parquet'); print('rows', B.height)
