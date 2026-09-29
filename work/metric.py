import polars as pl
def macro_f05(pred, gt, n_s1):
    """pred/gt: DataFrames (r1, ro). Exact competition metric: per-S1 F0.5, macro over all S1."""
    tp = pred.join(gt, on=['r1','ro']).group_by('r1').len('tp')
    npred = pred.group_by('r1').len('np'); ntrue = gt.group_by('r1').len('nt')
    d = ntrue.join(npred, on='r1', how='full', coalesce=True).join(tp, on='r1', how='left').fill_null(0)
    p = pl.col('tp')/pl.col('np'); r = pl.col('tp')/pl.col('nt')
    f = pl.when(pl.col('tp')==0).then(0.0).otherwise(1.25*p*r/(0.25*p+r))
    s = d.select(f.alias('f'))['f'].sum()
    # S1s with no true and no predicted matches score 1.0
    n_touched = d.height
    return (s + (n_s1 - n_touched)) / n_s1
