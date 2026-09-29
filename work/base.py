import polars as pl, sys; sys.path.insert(0,'work'); from metric import macro_f05
gt=pl.read_parquet('work/train/gt.parquet'); n_s1=2206821
P=pl.read_parquet('work/train/cand.parquet').filter(pl.col('rk')==1)
for t in [0,10,20,30,40,60]:
    print(t, round(macro_f05(P.filter(pl.col('score')>t).select('r1','ro'),gt,n_s1),4), flush=True)
