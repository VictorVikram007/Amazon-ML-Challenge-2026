import polars as pl
K=(pl.col('ro').cast(pl.UInt64)*2**32+pl.col('r1')).alias('k')
gt=pl.read_parquet('work/train/gt.parquet')
F=pl.read_parquet('work/train/oof.parquet')
best=F.sort('p',descending=True).unique('ro',keep='first').select('ro',rb='r1',pb='p'); 
fk=F.select(K,'p'); del F
pred=best.filter(pl.col('pb')>0.6)
g=gt.select('ro','r1',K).join(fk,on='k',how='left').join(best,on='ro',how='left')
tp=g.filter(pl.col('rb')==pl.col('r1'),pl.col('pb')>0.6).height
print('pred',pred.height,'TP',tp,'FP',pred.height-tp,'true',gt.height,'FN',gt.height-tp)
print('FN not in top3:',g['p'].is_null().sum())
print('FN other S1 won:',g.filter(pl.col('p').is_not_null(),pl.col('rb')!=pl.col('r1'),pl.col('pb')>0.6).height)
print('FN below thr:',g.filter(pl.col('p').is_not_null(),pl.col('rb')==pl.col('r1'),pl.col('pb')<=0.6).height)
print('FN other S1 top but below thr:',g.filter(pl.col('p').is_not_null(),pl.col('rb')!=pl.col('r1'),pl.col('pb')<=0.6).height)
fp=pred.join(gt.select('ro',rt='r1'),on='ro',how='left').filter(pl.col('rt').is_null() | (pl.col('rt')!=pl.col('rb')))
print('FP unmatched-in-truth:',fp['rt'].is_null().sum(),'FP belongs elsewhere:',fp['rt'].is_not_null().sum())
