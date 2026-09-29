"""Per-S1 expected-F0.5-optimal subset selection from calibrated match probabilities."""
import polars as pl, sys
sys.path.insert(0,'work'); from metric import macro_f05
def select(R, c=0.0, qmin=0.05, a=1.0):
    """R: (r1, ro, q). For each S1 pick top-k maximizing E[F0.5] ~ 1.25*TP_k/(k + 0.25*(NT+c)),
       vs. predicting nothing, worth P(no true match) = prod(1-q)."""
    x = (R.filter(pl.col('q') > qmin).with_columns(q=pl.col('q')**a).sort(['r1','q'], descending=[False,True])
          .with_columns(k=pl.int_range(1, pl.len()+1).over('r1'), tp=pl.col('q').cum_sum().over('r1'),
                        nt=pl.col('q').sum().over('r1'), p0=(1-pl.col('q')).log().sum().over('r1').exp()))
    x = x.with_columns(ef=1.25*pl.col('tp')/(pl.col('k')+0.25*(pl.col('nt')+c)))
    x = x.with_columns(best=pl.col('ef').max().over('r1'))
    kbest = x.filter(pl.col('ef')==pl.col('best')).group_by('r1').agg(kb=pl.col('k').min(), best=pl.col('best').first(), p0=pl.col('p0').first())
    x = x.join(kbest.select('r1','kb','p0',b2='best'), on='r1').filter(pl.col('k')<=pl.col('kb'), pl.col('b2')>pl.col('p0'))
    return x.select('r1','ro')
if __name__ == '__main__':
    R = pl.read_parquet('work/train/oof2.parquet'); gt = pl.read_parquet('work/train/gt.parquet')
    N1 = pl.read_parquet('work/train/s1.parquet', columns=['rid']).height
    print('global thr 0.7:', round(macro_f05(R.filter(pl.col('q')>0.7).select('r1','ro'), gt, N1), 5), flush=True)
    for c in [0.0, 0.3, 0.6]:
        for a in [1.0, 1.5]:
            print('expF c',c,'a',a, round(macro_f05(select(R, c=c, a=a), gt, N1), 5), flush=True)
