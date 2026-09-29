import polars as pl, lightgbm as lgb, glob, joblib
gt=pl.read_parquet('work/train/gt.parquet').with_columns(y=pl.lit(1,pl.UInt8))
files=sorted(glob.glob('work/train/feat/*.parquet'))
S=pl.concat([pl.read_parquet(f).sample(fraction=0.11,seed=i).join(gt,on=['r1','ro'],how='left').with_columns(pl.col('y').fill_null(0)) for i,f in enumerate(files)])
FEATS=[c for c in S.columns if c not in ('r1','ro','y')]; joblib.dump(FEATS,'work/feats_list.pkl')
prm=dict(objective='binary',learning_rate=0.08,num_leaves=255,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=12)
for k in [0,1]:
    tr=S.filter(pl.col('r1')%2!=k)
    lgb.train(prm,lgb.Dataset(tr.select(pl.col(FEATS).cast(pl.Float32)).to_numpy(),tr['y'].to_numpy()),num_boost_round=500).save_model(f'work/model_{k}.txt')
print('saved',len(FEATS))
