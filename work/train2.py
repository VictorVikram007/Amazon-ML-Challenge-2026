"""Stage-2 (group-level) model, memory-lean: sample before densifying, predict in chunks."""
import polars as pl, lightgbm as lgb, numpy as np, sys, gc
sys.path.insert(0,'work'); from metric import macro_f05
N1=pl.read_parquet('work/train/s1.parquet',columns=['rid']).height
gt=pl.read_parquet('work/train/gt.parquet')
B=pl.read_parquet('work/train/s2.parquet').join(gt.with_columns(y=pl.lit(1,pl.UInt8)),on=['r1','ro'],how='left').with_columns(pl.col('y').fill_null(0))
FE=[c for c in B.columns if c not in ('r1','ro','y')]
prm=dict(objective='binary',learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=10)
q=np.zeros(B.height,np.float32); fo=(B['r1']%2).to_numpy()
for k in [0,1]:
    tr=B.filter(pl.col('r1')%2!=k).sample(n=min(2_000_000,(fo!=k).sum()),seed=k)
    X=tr.select(pl.col(FE).cast(pl.Float32)).to_numpy(); y=tr['y'].to_numpy(); del tr; gc.collect()
    m=lgb.train(prm,lgb.Dataset(X,y,free_raw_data=True),num_boost_round=600); del X; gc.collect()
    m.save_model(f'work/model2_{k}.txt')
    idx=np.where(fo==k)[0]
    for i in range(0,len(idx),1_000_000):
        j=idx[i:i+1_000_000]; q[j]=m.predict(B[j].select(pl.col(FE).cast(pl.Float32)).to_numpy())
    print('fold',k,'saved',flush=True)
R=B.select('r1','ro','p').with_columns(q=q); R.write_parquet('work/train/oof2.parquet'); del B; gc.collect()
for t in [0.6,0.7]: print('stage1',t,round(macro_f05(R.filter(pl.col('p')>t).select('r1','ro'),gt,N1),4))
for t in [0.5,0.6,0.65,0.7,0.8,0.9,0.95]: print('stage2',t,round(macro_f05(R.filter(pl.col('q')>t).select('r1','ro'),gt,N1),4),flush=True)
