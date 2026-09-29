import polars as pl, sys
f,c=sys.argv[1],sys.argv[2]
K=(pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')).alias('k')
ro_all=pl.read_parquet('work/train/so.parquet',columns=['rid','country']).filter(pl.col('country')==c)['rid']
gt=pl.read_parquet('work/train/gt.parquet').filter(pl.col('ro').is_in(ro_all.implode())).select(K)
P=pl.scan_parquet(f'work/train/{f}').select(K,'rk').join(gt.lazy(),on='k',how='semi').collect(engine='streaming')
m=gt.join(P,on='k',how='left')['rk']
print(f,'n',gt.height,{k: round((m<=k).fill_null(False).mean(),4) for k in [1,3,5,10,20,50]})
