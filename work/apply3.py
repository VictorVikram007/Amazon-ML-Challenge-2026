import polars as pl, lightgbm as lgb, numpy as np, sys, os
FE = ['q', 's1sim', 's1n', 'h_eq', 'n_sib', 'sib_same_h', 'sib_h1', 'best_sib', 'best_sibn', 'n_closer', 'n_closer_n']
S = pl.read_parquet('work/test/sib.parquet'); X = S.select(pl.col(FE).cast(pl.Float32)).to_numpy()
M = [lgb.Booster(model_file=f'work/model3_{k}.txt') for k in [0, 1]]
S.select('r1', 'ro').with_columns(q=((M[0].predict(X) + M[1].predict(X)) / 2).astype(np.float32)).write_parquet('work/test/q3.parquet')
print('scored', S.height)
