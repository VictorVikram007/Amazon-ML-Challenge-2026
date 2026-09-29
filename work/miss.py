import polars as pl, sys
sys.stdout.reconfigure(encoding='utf-8')
K=lambda: (pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')).alias('key')
gt=pl.read_parquet('work/train/gt.parquet').sample(200000,seed=3).with_columns(K())
P=pl.scan_parquet('work/train/cand.parquet').select(K()).collect()
m=gt.filter(~pl.col('key').is_in(P['key'].implode())); del P
print('missing frac',m.height/gt.height); m=m.head(40)
s1=pl.scan_parquet('work/train/s1.parquet').filter(pl.col('rid').is_in(m['r1'].implode())).select(r1='rid',n1='business_name',a1='business_address').collect()
so=pl.scan_parquet('work/train/so.parquet').filter(pl.col('rid').is_in(m['ro'].implode())).select(ro='rid',no='business_name',ao='business_address').collect()
for r in m.join(s1,on='r1').join(so,on='ro').iter_rows(named=True): print(r['n1'],'|',r['a1'],'\n    ',r['no'],'|',r['ao'])
