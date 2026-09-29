import polars as pl
for c in ['us','india']:
    T1=pl.read_parquet(f'work/train/tok_1_{c}.parquet'); To=pl.read_parquet(f'work/train/tok_o_{c}.parquet')
    d=T1.group_by('tok').len('d1').join(To.group_by('tok').len('do'),on='tok')
    print(c,'rows',T1.height,To.height)
    for cap in [10,30,100,300,1000]:
        x=d.filter(pl.col('d1')<=cap)
        print(' cap',cap,'pairs(M)=',round((x['d1'].cast(pl.Int64)*x['do']).sum()/1e6,1),'toks',x.height,'/',d.height)
