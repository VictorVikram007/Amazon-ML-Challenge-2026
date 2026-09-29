import polars as pl, numpy as np, sys, time
from rapidfuzz import process, fuzz
sys.path.insert(0,'work'); from featx import GEN, fixtok
core=lambda s: ' '.join(sorted(t2 for t in s.split() if len(t)>1 and (t2:=fixtok(t)) not in GEN))
K=(pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')).alias('k')
so=pl.read_parquet('work/train/so.parquet',columns=['rid','country','nm','ad']).filter(pl.col('ad').str.len_chars()<3,~pl.col('nm').str.contains(r'[^\x00-\x7f]'))
C=pl.read_parquet('work/train/cand.parquet',columns=['ro','r1','rk']).filter(pl.col('rk')<=5).select(K)
gt=pl.read_parquet('work/train/gt.parquet').join(so.select(ro='rid',country='country',no='nm'),on='ro').with_columns(K)
miss=gt.filter(~pl.col('k').is_in(C['k'].implode())).sample(1500,seed=0)
s1=pl.read_parquet('work/train/s1.parquet',columns=['rid','country','nm'])
res=[]
for c in ['us','india']:
    S=s1.filter(pl.col('country')==c); names=[core(x) for x in S['nm'].to_list()]; rid=S['rid'].to_numpy(); pos={r:i for i,r in enumerate(rid)}
    M=miss.filter(pl.col('country')==c); t=time.time()
    qs=[core(x) for x in M['no'].to_list()]; r1s=M['r1'].to_list()
    for i in range(0,len(qs),100):
        D=process.cdist(qs[i:i+100],names,scorer=fuzz.token_sort_ratio,dtype=np.uint8,workers=-1)
        for j,row in enumerate(D):
            tv=row[pos[r1s[i+j]]]; res.append((c,int(1+(row>tv).sum()),int((row==tv).sum()),int(tv)))
    print(c,'time',round(time.time()-t))
R=pl.DataFrame(res,schema=['c','rank','ties','tv'],orient='row')
print(R.group_by('c').agg(n=pl.len(),top1_unique=((pl.col('rank')==1)&(pl.col('ties')==1)).mean(),top1=(pl.col('rank')==1).mean(),top5=(pl.col('rank')<=5).mean(),tv_med=pl.col('tv').median()))
