import polars as pl, glob
d='work/train/tok_1_india'
T1=pl.read_parquet(f'{d}/*.parquet'); print('T1 rows',T1.height,flush=True)
df=T1.group_by('tok').len('d1'); print('distinct',df.height, flush=True)
for cap in [10,30,100]:
    x=df.filter(pl.col('d1')<=cap); print(cap,'kept rows',x['d1'].sum(),flush=True)
ch=pl.read_parquet(sorted(glob.glob('work/train/tok_o_india/*.parquet'))[0]); print('chunk rows',ch.height,ch['rid'].n_unique())
j=ch.join(df,on='tok')
for cap in [10,30,100]: print(cap,'join rows(M)',j.filter(pl.col('d1')<=cap)['d1'].cast(pl.Int64).sum()/1e6)
