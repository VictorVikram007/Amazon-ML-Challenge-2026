import polars as pl, numpy as np, time
from rapidfuzz import process, fuzz
cp=lambda a,b,sc: process.cpdist(a,b,scorer=sc,workers=-1).astype(np.float32)
so_ids=pl.read_parquet('work/train/so.parquet',columns=['rid','country']).filter(pl.col('country')=='india')['rid'].sample(300000,seed=1)
P=pl.scan_parquet('work/train/cand_i50.parquet').filter(pl.col('ro').is_in(so_ids.implode())).collect()
s1=pl.read_parquet('work/train/s1.parquet',columns=['rid','nm','ad','sk']).rename({'rid':'r1','nm':'n1','ad':'a1','sk':'k1'})
so=pl.read_parquet('work/train/so.parquet',columns=['rid','nm','ad','sk']).filter(pl.col('rid').is_in(so_ids.implode())).rename({'rid':'ro','nm':'no','ad':'ao','sk':'ko'})
x=P.join(s1,on='r1').join(so,on='ro'); t=time.time()
x=x.with_columns(kr=cp(x['k1'].to_list(),x['ko'].to_list(),fuzz.token_set_ratio),nr=cp(x['n1'].to_list(),x['no'].to_list(),fuzz.token_set_ratio),
                 ar=cp(x['a1'].to_list(),x['ao'].to_list(),fuzz.token_set_ratio)).select('ro','r1','rk','score','kr','nr','ar')
print('pairs',x.height,'time',round(time.time()-t))
gt=pl.read_parquet('work/train/gt.parquet').filter(pl.col('ro').is_in(so_ids.implode())).with_columns(y=pl.lit(1))
x=x.join(gt,on=['ro','r1'],how='left').with_columns(pl.col('y').fill_null(0))
x.write_parquet('work/train/rerank_sample.parquet')
n=gt.height
for name,expr in [('block',pl.col('score')),('nm+ad',pl.max_horizontal('kr','nr')+pl.col('ar')),('nm+ad+blk',pl.max_horizontal('kr','nr')+pl.col('ar')+2*pl.col('score')),
                  ('nm*ad',pl.max_horizontal('kr','nr')*(pl.col('ar')+30))]:
    r=x.with_columns(rr=expr.rank('ordinal',descending=True).over('ro'))
    print(name,{k: round(r.filter(pl.col('y')==1,pl.col('rr')<=k).height/n,4) for k in [1,3,5,10]})
