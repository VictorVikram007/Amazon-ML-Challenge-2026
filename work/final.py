import polars as pl, lightgbm as lgb, numpy as np, glob, gc, sys, os
T=float(sys.argv[1]) if len(sys.argv)>1 else 0.6
gt=pl.read_parquet('work/train/gt.parquet').with_columns(y=pl.lit(1,pl.UInt8))
files=sorted(glob.glob('work/train/feat/*.parquet'))
S=pl.concat([pl.read_parquet(f).sample(fraction=0.11,seed=i).join(gt,on=['r1','ro'],how='left').with_columns(pl.col('y').fill_null(0)) for i,f in enumerate(files)])
FEATS=[c for c in S.columns if c not in ('r1','ro','y')]
prm=dict(objective='binary',learning_rate=0.08,num_leaves=255,min_data_in_leaf=200,feature_fraction=0.8,
         bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=12)
models=[]
for k in [0,1]:
    tr=S.filter(pl.col('r1')%2!=k)
    m=lgb.train(prm,lgb.Dataset(tr.select(pl.col(FEATS).cast(pl.Float32)).to_numpy(),tr['y'].to_numpy()),num_boost_round=500)
    m.save_model(f'work/model_{k}.txt'); models.append(m)
del S; gc.collect(); print('trained',flush=True)
outs=[]
for f in sorted(glob.glob('work/test/feat/*.parquet')):
    d=pl.read_parquet(f); X=d.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()
    p=(models[0].predict(X)+models[1].predict(X))/2
    outs.append(d.select('r1','ro').with_columns(p=p.astype(np.float32))); del d,X; gc.collect()
F=pl.concat(outs); F.write_parquet('work/test/pred.parquet')
best=F.sort('p',descending=True).unique('ro',keep='first').filter(pl.col('p')>T)
cand=F.filter(pl.col('p')>0.05).select('r1','ro')
s1=pl.read_parquet('work/test/s1.parquet',columns=['rid','entity_id']).rename({'rid':'r1','entity_id':'source1_entity_id'})
so=pl.read_parquet('work/test/so.parquet',columns=['rid','entity_id']).rename({'rid':'ro','entity_id':'id'})
def write(pairs,col,path):
    g=pairs.join(so,on='ro').group_by('r1').agg(pl.col('id').sort().str.join(',').alias(col))
    s1.join(g,on='r1',how='left').with_columns(pl.col(col).fill_null('')).select('source1_entity_id',col) \
      .write_csv(path,separator='\t',quote_style='never')
os.makedirs('submission',exist_ok=True)
write(best.select('r1','ro'),'matched_entity_ids','submission/matching_results.tsv')
write(pl.concat([cand,best.select('r1','ro')]).unique(),'candidate_entity_ids','submission/candidate_pairs.tsv')
print('matched pairs',best.height,'S1 with matches',best['r1'].n_unique(),'of',s1.height)
