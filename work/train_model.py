import polars as pl, lightgbm as lgb, numpy as np, sys, gc, glob, joblib
sys.path.insert(0,'work'); from metric import macro_f05
N1=pl.read_parquet('work/train/s1.parquet',columns=['rid']).height
gt=pl.read_parquet('work/train/gt.parquet').with_columns(y=pl.lit(1,pl.UInt8))
files=sorted(glob.glob('work/train/feat/*.parquet'))
lab=lambda d: d.join(gt,on=['r1','ro'],how='left').with_columns(pl.col('y').fill_null(0))
S=pl.concat([lab(pl.read_parquet(f).sample(fraction=0.11,seed=i)) for i,f in enumerate(files)])
FEATS=[c for c in S.columns if c not in ('r1','ro','y')]
print('sample',S.height,'pos',round(S['y'].mean(),3),'feats',len(FEATS),flush=True)
prm=dict(objective='binary',learning_rate=0.08,num_leaves=255,min_data_in_leaf=200,feature_fraction=0.8,
         bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=12)
models=[]
for k in [0,1]:
    tr=S.filter(pl.col('r1')%2!=k)
    models.append(lgb.train(prm,lgb.Dataset(tr.select(pl.col(FEATS).cast(pl.Float32)).to_numpy(),tr['y'].to_numpy()),num_boost_round=500))
    models[-1].save_model(f'work/model_{k}.txt')
    print('fold',k,flush=True)
del S; gc.collect()
imp=sorted(zip(models[0].feature_importance('gain'),FEATS),reverse=True); print([f for _,f in imp[:12]],flush=True)
outs=[]
for f in files:
    d=pl.read_parquet(f); p=np.zeros(d.height,np.float32); fo=(d['r1']%2).to_numpy()
    X=d.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()
    for k in [0,1]: p[fo==k]=models[k].predict(X[fo==k])
    outs.append(d.select('r1','ro').with_columns(p=p)); del d,X; gc.collect()
F=pl.concat(outs); F.write_parquet('work/train/oof.parquet')
best=F.sort('p',descending=True).unique('ro',keep='first')
g=gt.select('r1','ro')
for t in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    print(t, round(macro_f05(best.filter(pl.col('p')>t).select('r1','ro'),g,N1),4),flush=True)
joblib.dump(FEATS,'work/feats_list.pkl')
