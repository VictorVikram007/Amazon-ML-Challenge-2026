"""Stage 3: recalibrate stage-2 confidence with phantom-cluster / sibling features (cross-fitted by S1)."""
import polars as pl, lightgbm as lgb, numpy as np, sys
sys.path.insert(0, 'work'); from metric import macro_f05
FE = ['q', 's1sim', 's1n', 'h_eq', 'n_sib', 'sib_same_h', 'sib_h1', 'best_sib', 'best_sibn', 'n_closer', 'n_closer_n']
prm = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, verbose=-1, num_threads=12)
gt = pl.read_parquet('work/train/gt.parquet')
S = pl.read_parquet('work/train/sib.parquet').join(gt.with_columns(y=pl.lit(1, pl.UInt8)), on=['r1', 'ro'], how='left').with_columns(pl.col('y').fill_null(0))
N1 = pl.read_parquet('work/train/s1.parquet', columns=['rid']).height
X = S.select(pl.col(FE).cast(pl.Float32)).to_numpy(); y = S['y'].to_numpy(); fo = (S['r1'] % 2).to_numpy(); z = np.zeros(len(y), np.float32)
for k in [0, 1]:
    m = lgb.train(prm, lgb.Dataset(X[fo != k], y[fo != k]), 300); z[fo == k] = m.predict(X[fo == k]); m.save_model(f'work/model3_{k}.txt')
R = S.select('r1', 'ro', 'q').with_columns(z=z)
for t in [0.7, 0.85]: print('stage2 q', t, round(macro_f05(R.filter(pl.col('q') > t).select('r1', 'ro'), gt, N1), 5))
for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]: print('stage3 z', t, round(macro_f05(R.filter(pl.col('z') > t).select('r1', 'ro'), gt, N1), 5), flush=True)
