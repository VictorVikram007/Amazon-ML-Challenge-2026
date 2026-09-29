"""Phantom-cluster features: does a record have siblings (other records whose best S1 is the same)
that agree with IT rather than with S1? Decoys come as consistent clusters of a phantom business;
a true match with a corrupted field is a one-off."""
import polars as pl, numpy as np, sys, gc
from rapidfuzz import process, fuzz

split, qfile = sys.argv[1], sys.argv[2]
cp = lambda a, b: process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
fnum = lambda c: pl.col(c).str.extract(r'(\d+)')

RR = pl.read_parquet(f'work/{split}/{qfile}').filter(pl.col('q') > 0.05).select('r1', 'ro', 'q')
s1 = pl.read_parquet(f'work/{split}/s1.parquet', columns=['rid', 'nm', 'ad']).rename({'rid': 'r1', 'nm': 'n1', 'ad': 'a1'})
so = pl.read_parquet(f'work/{split}/so.parquet', columns=['rid', 'nm', 'ad']).rename({'rid': 'ro', 'nm': 'no', 'ad': 'ao'})
COLS = ['r1', 'ro', 'q', 's1sim', 's1n', 'h_eq', 'n_sib', 'sib_same_h', 'sib_h1', 'best_sib', 'best_sibn', 'n_closer', 'n_closer_n']
outs, STEP = [], 150_000
for lo in range(0, RR['r1'].max() + 1, STEP):
    X = (RR.filter(pl.col('r1').is_between(lo, lo + STEP - 1)).join(so, on='ro').join(s1, on='r1')
         .with_columns(hx=fnum('ao'), h1=fnum('a1'), fx=pl.col('no') + ' | ' + pl.col('ao'), f1=pl.col('n1') + ' | ' + pl.col('a1')))
    X = X.with_columns(s1sim=cp(X['fx'].to_list(), X['f1'].to_list()), s1n=cp(X['no'].to_list(), X['n1'].to_list()))
    P = (X.select('r1', 'ro', 'hx', 'fx', 'no')
         .join(X.select('r1', ro2='ro', hx2='hx', fx2='fx', no2='no'), on='r1').filter(pl.col('ro') != pl.col('ro2')))
    P = P.with_columns(ss=cp(P['fx'].to_list(), P['fx2'].to_list()), sn=cp(P['no'].to_list(), P['no2'].to_list()))
    G = P.join(X.select('ro', 's1sim', 's1n', 'h1'), on='ro').group_by('ro').agg(
        n_sib=pl.len(),
        sib_same_h=((pl.col('hx') == pl.col('hx2')) & (pl.col('hx') != pl.col('h1'))).sum(),  # siblings backing X's own (non-S1) number
        sib_h1=(pl.col('hx2') == pl.col('h1')).sum(),                                           # siblings backing S1's number
        best_sib=pl.col('ss').max(), best_sibn=pl.col('sn').max(),
        n_closer=(pl.col('ss') > pl.col('s1sim')).sum(), n_closer_n=(pl.col('sn') > pl.col('s1n')).sum())
    X = X.join(G, on='ro', how='left').with_columns(
        pl.col('n_sib', 'sib_same_h', 'sib_h1', 'n_closer', 'n_closer_n').fill_null(0),
        pl.col('best_sib', 'best_sibn').fill_null(0), h_eq=(pl.col('hx') == pl.col('h1')).fill_null(False))
    outs.append(X.select(COLS)); del X, P, G; gc.collect()
    print(lo, flush=True)
out = pl.concat(outs); out.write_parquet(f'work/{split}/sib.parquet'); print('rows', out.height)
