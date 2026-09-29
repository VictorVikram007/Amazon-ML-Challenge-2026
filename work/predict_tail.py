import polars as pl, lightgbm as lgb, numpy as np, glob, gc, joblib, subprocess, sys, os
T=float(sys.argv[1]); OUTDIR=sys.argv[2]
FEATS=joblib.load('work/feats_list.pkl'); M=[lgb.Booster(model_file=f'work/model_{k}.txt') for k in [0,1]]
F=pl.read_parquet('work/test/pred.parquet')
B=pl.read_parquet('work/test/s2.parquet'); M2=[lgb.Booster(model_file=f'work/model2_{k}.txt') for k in [0,1]]
FE=[c for c in B.columns if c not in ('r1','ro')]; X=B.select(pl.col(FE).cast(pl.Float32)).to_numpy()
B=B.select('r1','ro').with_columns(q=((M2[0].predict(X)+M2[1].predict(X))/2).astype(np.float32))
B.write_parquet('work/test/q.parquet')
best=B.filter(pl.col('q')>T).select('r1','ro')
cand=pl.concat([F.filter(pl.col('p')>0.05).select('r1','ro'),best]).unique()
s1=pl.read_parquet('work/test/s1.parquet',columns=['rid','entity_id']).rename({'rid':'r1','entity_id':'source1_entity_id'})
so=pl.read_parquet('work/test/so.parquet',columns=['rid','entity_id']).rename({'rid':'ro','entity_id':'id'})
def write(pairs,col,path):
    g=pairs.join(so,on='ro').group_by('r1').agg(pl.col('id').sort().str.join(',').alias(col))
    s1.join(g,on='r1',how='left').with_columns(pl.col(col).fill_null('')).select('source1_entity_id',col).write_csv(path,separator='\t',quote_style='never')
os.makedirs(OUTDIR,exist_ok=True)
write(best,'matched_entity_ids',OUTDIR+'/matching_results.tsv')
write(cand,'candidate_entity_ids',OUTDIR+'/candidate_pairs.tsv')
print('matched pairs',best.height,'S1 with matches',best['r1'].n_unique(),'of',s1.height)
