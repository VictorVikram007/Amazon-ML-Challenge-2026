import polars as pl, sys, os
T=float(sys.argv[1]); OUT=sys.argv[2]
Q=pl.read_parquet('work/test/q3.parquet'); F=pl.read_parquet('work/test/pred.parquet')
best=Q.filter(pl.col('q')>T).select('r1','ro').unique()
cand=pl.concat([F.filter(pl.col('p')>0.05).select('r1','ro'),best]).unique()
s1=pl.read_parquet('work/test/s1.parquet',columns=['rid','entity_id']).rename({'rid':'r1','entity_id':'source1_entity_id'})
so=pl.read_parquet('work/test/so.parquet',columns=['rid','entity_id']).rename({'rid':'ro','entity_id':'id'})
def write(pairs,col,path):
    g=pairs.join(so,on='ro').group_by('r1').agg(pl.col('id').unique().sort().str.join(',').alias(col))
    s1.join(g,on='r1',how='left').with_columns(pl.col(col).fill_null('')).select('source1_entity_id',col).write_csv(path,separator='\t',quote_style='never')
os.makedirs(OUT,exist_ok=True)
write(best,'matched_entity_ids',f'{OUT}/matching_results.tsv'); write(cand,'candidate_entity_ids',f'{OUT}/candidate_pairs.tsv')
print(OUT,'pairs',best.height)
