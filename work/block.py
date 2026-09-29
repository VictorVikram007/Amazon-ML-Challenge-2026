"""Candidates for every S2/S3 record: top-50 S1 by IDF-weighted shared keys, re-ranked by fuzzy name/address scores, keep top-KEEP."""
import polars as pl, sys, gc, time, glob, numpy as np
from rapidfuzz import process, fuzz
split=sys.argv[1]; CAP=int(sys.argv[2]); K=int(sys.argv[3]); KEEP=int(sys.argv[4]); NSUB=int(sys.argv[5]) if len(sys.argv)>5 else 4
cp=lambda a,b,sc: process.cpdist(a,b,scorer=sc,workers=-1).astype(np.float32)
s1=pl.read_parquet(f'work/{split}/s1.parquet',columns=['rid','nm','ad','sk']).rename({'rid':'r1','nm':'n1','ad':'a1','sk':'k1'})
so=pl.read_parquet(f'work/{split}/so.parquet',columns=['rid','nm','ad','sk']).rename({'rid':'ro','nm':'no','ad':'ao','sk':'ko'})
import os
PD=f'work/{split}/cand_parts'; os.makedirs(PD,exist_ok=True); nout=0
for d in sorted(glob.glob(f'work/{split}/tok_1_*')):
    c=d.split('tok_1_')[1]; t=time.time()
    if os.path.exists(f'{PD}/{c}.done'): continue
    T1=pl.read_parquet(f'{d}/*.parquet'); n1=T1['rid'].n_unique()
    df=T1.group_by('tok').len('d1').filter(pl.col('d1')<=CAP).with_columns(w=(n1/pl.col('d1')).log().cast(pl.Float32))
    T1=T1.join(df.select('tok','w'),on='tok'); del df; gc.collect()
    X1=pl.read_parquet(f'work/{split}/xtok_1_{c}.parquet')
    X1=X1.join(X1.group_by('tok').len('d1').filter(pl.col('d1')<=500).with_columns(w=(n1/pl.col('d1')).log().cast(pl.Float32)).select('tok','w'),on='tok')
    Xo=pl.read_parquet(f'work/{split}/xtok_o_{c}.parquet')
    for f in sorted(glob.glob(f'work/{split}/tok_o_{c}/*.parquet')):
      full=pl.read_parquet(f); r=full['rid']; lo,hi=r.min(),r.max(); st=(hi-lo)//NSUB+1
      for q in range(NSUB):
        ch=full.filter(pl.col('rid').is_between(lo+q*st,lo+(q+1)*st-1))
        xo=Xo.filter(pl.col('rid').is_between(lo+q*st,lo+(q+1)*st-1))
        j=pl.concat([ch.join(T1,on='tok',suffix='_1'),xo.join(X1,on='tok',suffix='_1')])
        p=(j.group_by('rid','rid_1').agg(score=pl.col('w').sum(),nt=pl.len().cast(pl.UInt16))
             .with_columns(brk=pl.col('score').rank('ordinal',descending=True).over('rid'))
             .filter(pl.col('brk')<=K).rename({'rid':'ro','rid_1':'r1'}))
        del ch,j,xo; gc.collect()
        x=p.join(s1,on='r1').join(so,on='ro')
        x=x.select('ro','r1','score','nt','brk',kr=cp(x['k1'].to_list(),x['ko'].to_list(),fuzz.token_set_ratio),
                   nr=cp(x['n1'].to_list(),x['no'].to_list(),fuzz.token_set_ratio),ar=cp(x['a1'].to_list(),x['ao'].to_list(),fuzz.token_set_ratio))
        x=(x.with_columns(rs=pl.max_horizontal('kr','nr')+pl.col('ar')+2*pl.col('score'))
            .with_columns(rk=pl.col('rs').rank('ordinal',descending=True).over('ro')).filter(pl.col('rk')<=KEEP)
            .with_columns(pl.col('rk').cast(pl.UInt8),pl.col('brk').cast(pl.UInt8)))
        x.write_parquet(f'{PD}/{c}_{nout:05d}.parquet'); nout+=1; del p,x; gc.collect()
      del full
    open(f'{PD}/{c}.done','w').close()
    print(c,'done',round(time.time()-t),'s',flush=True); del T1,X1,Xo; gc.collect()
if len(sys.argv)>6 and sys.argv[6]=='nomerge': sys.exit(0)
pl.scan_parquet(f'{PD}/*.parquet').sink_parquet(f'work/{split}/cand.parquet'); import shutil; shutil.rmtree(PD); print('cand written')
