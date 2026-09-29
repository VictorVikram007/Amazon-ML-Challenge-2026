import polars as pl, sys, os, gc, time, numpy as np
from rapidfuzz import process, fuzz, distance
split=sys.argv[1]; TOPK=int(sys.argv[2]) if len(sys.argv)>2 else 3
out=f'work/{split}/feat'; os.makedirs(out,exist_ok=True)
RESUME=os.path.exists(f'work/{split}/_p.parquet')
C=pl.scan_parquet(f'work/{split}/cand.parquet')
if not RESUME:
  RO=C.group_by('ro').agg(top=pl.col('score').max(),ncand=pl.len().cast(pl.UInt8),
                          s2=pl.col('score').sort(descending=True).get(1,null_on_oob=True)).collect(engine='streaming')
  P=C.filter(pl.col('rk')<=TOPK).collect(engine='streaming').join(RO,on='ro').with_columns(pl.col('s2').fill_null(0))
  del RO; gc.collect()
  P=P.with_columns(gap=pl.col('score')-pl.when(pl.col('rk')==1).then('s2').otherwise('top'),rel=pl.col('score')/pl.col('top')).drop('s2')
  R1=P.group_by('r1').agg(n1_top=(pl.col('rk')==1).sum().cast(pl.UInt16),n1_all=pl.len().cast(pl.UInt16))
  P=P.join(R1,on='r1'); del R1; gc.collect()
  P=P.with_columns(rk1=pl.col('score').rank('ordinal',descending=True).over('r1').cast(pl.UInt16)).sort('ro')
  NP=P.height; P.write_parquet(f'work/{split}/_p.parquet'); del P; gc.collect();   print('pairs',NP,flush=True)
else:
  NP=pl.scan_parquet(f'work/{split}/_p.parquet').select(pl.len()).collect().item()
s1=pl.read_parquet(f'work/{split}/s1.parquet',columns=['rid','nm','ad','sk']).rename({'rid':'r1','nm':'n1','ad':'a1','sk':'k1'})
so=pl.read_parquet(f'work/{split}/so.parquet',columns=['rid','entity_id','nm','ad','sk']).select(
    ro='rid',no='nm',ao='ad',ko='sk',src3=pl.col('entity_id').str.starts_with('S3'))
nums=lambda c: pl.col(c).str.extract_all(r'\d+').list.unique()
def cp(a,b,sc): return process.cpdist(a,b,scorer=sc,workers=-1).astype(np.float32)
CH=1_000_000
for j,i in enumerate(range(0,NP,CH)):
    t=time.time()
    if os.path.exists(f'{out}/{j:04d}.parquet'): continue
    x=pl.scan_parquet(f'work/{split}/_p.parquet').slice(i,CH).collect().join(s1,on='r1').join(so,on='ro')
    nc=lambda c: pl.col(c).str.replace_all(' ','')
    x=x.with_columns(
        miss_a=(pl.col('ao').str.len_chars()==0), nonlat1=pl.col('n1').str.contains(r'[^\x00-\x7f]'),
        nonlato=pl.col('no').str.contains(r'[^\x00-\x7f]'),
        len_n1=pl.col('n1').str.len_chars().cast(pl.UInt16), len_no=pl.col('no').str.len_chars().cast(pl.UInt16),
        num_i=nums('a1').list.set_intersection(nums('ao')).list.len().cast(pl.UInt8),
        num_1=nums('a1').list.len().cast(pl.UInt8), num_o=nums('ao').list.len().cast(pl.UInt8),
        first_num_eq=(pl.col('a1').str.extract(r'(\d+)')==pl.col('ao').str.extract(r'(\d+)')).fill_null(False),
        c1=nc('n1'), co=nc('no'))
    n1,no,a1,ao,c1,co,k1,ko=[x[c].to_list() for c in ['n1','no','a1','ao','c1','co','k1','ko']]
    x=x.with_columns(
        n_ratio=cp(n1,no,fuzz.ratio), n_tset=cp(n1,no,fuzz.token_set_ratio), n_tsort=cp(n1,no,fuzz.token_sort_ratio),
        n_part=cp(n1,no,fuzz.partial_ratio), c_ratio=cp(c1,co,fuzz.ratio), c_part=cp(c1,co,fuzz.partial_ratio),
        c_jw=cp(c1,co,distance.JaroWinkler.normalized_similarity),
        a_ratio=cp(a1,ao,fuzz.ratio), a_tset=cp(a1,ao,fuzz.token_set_ratio), a_tsort=cp(a1,ao,fuzz.token_sort_ratio),
        a_part=cp(a1,ao,fuzz.partial_token_set_ratio),
        k_ratio=cp(k1,ko,fuzz.ratio), k_tset=cp(k1,ko,fuzz.token_set_ratio), k_tsort=cp(k1,ko,fuzz.token_sort_ratio),
    ).drop('n1','no','a1','ao','c1','co','k1','ko')
    x.write_parquet(f'{out}/{j:04d}.parquet'); del x,n1,no,a1,ao,c1,co,k1,ko; gc.collect()
    print(j, round(time.time()-t,1),'s',flush=True)

os.remove(f'work/{split}/_p.parquet')
