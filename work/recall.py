import polars as pl, gc, os
def eid(col):
    return (pl.col(col).str.slice(3).cast(pl.Int64) + pl.when(pl.col(col).str.starts_with('S3')).then(2**32).otherwise(0))
if not os.path.exists('work/train/gt.parquet'):
    s1=pl.read_parquet('work/train/s1.parquet',columns=['rid','entity_id']).select(r1='rid',k1=eid('entity_id'))
    so=pl.read_parquet('work/train/so.parquet',columns=['rid','entity_id']).select(ro='rid',ko=eid('entity_id'))
    gt=(pl.read_csv('dataset/train/train_ground_truth.tsv',separator='\t',infer_schema=False)
         .select(k1=eid('source1_entity_id'),m=pl.col('matched_entity_ids').str.split(',')).explode('m')
         .filter(pl.col('m').str.len_chars()>0).select('k1',ko=eid('m')))
    gt=gt.join(s1,on='k1').join(so,on='ko').select('r1','ro'); del s1,so; gc.collect()
    gt.write_parquet('work/train/gt.parquet'); print('gt written',gt.height,flush=True)
gt=pl.read_parquet('work/train/gt.parquet').select(key=(pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')))
P=pl.read_parquet('work/train/cand.parquet',columns=['ro','r1','rk']).select(key=(pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')),rk=pl.col('rk').cast(pl.UInt8))
print('loaded',flush=True)
m=gt.join(P,on='key',how='left')['rk']
print('true pairs',gt.height,'in cands',m.is_not_null().mean())
for k in [1,2,3,5,10]: print(' top',k,(m<=k).fill_null(False).mean())
