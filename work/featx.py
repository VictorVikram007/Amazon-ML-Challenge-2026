"""Difference-structure features: house-number relation, unexplained name/address words (filler-aware)."""
import polars as pl, numpy as np, sys, glob, re, time
from multiprocessing import Pool
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, 'work'); from norm import clean, ADDR_STOP
GEN = set("""com ltd center centre services service inc llc co corporation corp the partners lp limited smt incorporated mr mrs ms
shri sri dr dba formerly as www company llp and trading doing business aka nee enterprises fka known labs sys one pc trust board
foundation society association authority district commission council federation associates associate group pvt private pllc
india care clinic solutions of brothers technologies technology global industries holdings international ventures llc m s co
in id the ltd st smt kumari km late sons son""".split())
MARK = r'\b(?:t/a|d/b/a|dba|fka|f/k/a|aka|a/k/a|formerly|nee|doing business as|known as|trading as)\b'
STATES = set("""al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut
vt va wa wv wi wy alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa
kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada hampshire jersey mexico
york carolina dakota ohio oklahoma oregon pennsylvania rhode island tennessee texas utah vermont virginia washington wisconsin wyoming columbia
district new north south west""".split())
AST = set(ADDR_STOP) | STATES
OCR = str.maketrans('0158634', 'olsbgea')
def fixtok(t):
    return t.translate(OCR) if (any(c.isalpha() for c in t) and any(c.isdigit() for c in t)) else t
def ntoks(s):
    return [fixtok(t) for t in s.split() if len(t) > 1 and fixtok(t) not in GEN]
def unexplained(a, b, bcat):
    """count tokens of a with no close match in b (typo-tolerant, also allows match inside concatenated b)"""
    n = 0; worst = 100.0
    for t in a:
        best = max((fuzz.ratio(t, u) for u in b), default=0.0)
        if best < 75 and len(t) >= 4 and bcat and fuzz.partial_ratio(t, bcat) >= 85: best = 85.0
        worst = min(worst, best)
        if best < 67: n += 1
    return n, worst
def nums(s): return re.findall(r'\d+', s)
def row(args):
    n1, no, a1, ao = args
    t1, to = ntoks(n1), ntoks(no)
    ux_o, w_o = unexplained(to, t1, ''.join(t1)); ux_1, w_1 = unexplained(t1, to, ''.join(to))
    b1 = [t for t in a1.split() if len(t) >= 3 and not t.isdigit() and t not in AST]
    bo = [t for t in ao.split() if len(t) >= 3 and not t.isdigit() and t not in AST]
    ax_o, aw_o = unexplained(bo, b1, '') if b1 else (0, 100.0)
    ax_1, aw_1 = unexplained(b1, bo, '') if bo else (0, 100.0)
    m1, mo = nums(a1), nums(ao)
    h1, ho = (m1[0] if m1 else ''), (mo[0] if mo else '')
    if h1 and ho:
        hlev = Levenshtein.distance(h1, ho); hdiff = abs(int(h1[:9]) - int(ho[:9]))
        hsub = int(h1 != ho and (h1 in ho or ho in h1)); hlen = int(len(h1) == len(ho))
    else: hlev, hdiff, hsub, hlen = -1, -1, -1, -1
    so_ = set(mo); s1_ = set(m1)
    n1only = sum(1 for x in s1_ if x not in so_); noonly = sum(1 for x in so_ if x not in s1_)
    nnear = sum(1 for x in s1_ if x not in so_ and any(Levenshtein.distance(x, y) <= 1 for y in so_))
    # best house-number agreement: is ANY number of other exactly equal to h1
    hin = int(bool(h1) and h1 in so_)
    return (len(b1), len(bo), ux_o, w_o, ux_1, w_1, len(t1), len(to), ax_o, aw_o, ax_1, aw_1, hlev, min(hdiff, 10**6), hsub, hlen, n1only, noonly, nnear, hin)
COLS = ['nb1','nbo','ux_o','uw_o','ux_1','uw_1','nt1','nto','ax_o','aw_o','ax_1','aw_1','hlev','hdiff','hsub','hlen','n1only','noonly','nnear','hin']
def namef(col):  # strip alias prefixes ("X fka Y" -> Y) and "| website" suffixes, then normalize
    return clean('_n').alias(col) if False else None
if __name__ == '__main__':
    split = sys.argv[1]
    strip = lambda c: pl.col(c).fill_null('').str.to_lowercase().str.split('|').list.first().str.replace(r'^.*' + MARK, '')
    s1 = pl.read_parquet(f'work/{split}/s1.parquet', columns=['rid', 'business_name', 'ad']).select(
        r1='rid', a1='ad', _n=strip('business_name')).with_columns(n1=clean('_n')).drop('_n')
    pool = Pool(8)
    for f in sorted(glob.glob(f'work/{split}/feat/*.parquet')):
            t = time.time()
            if 'ux_o' in pl.read_parquet_schema(f): continue
            d = pl.read_parquet(f)
            so = (pl.scan_parquet(f'work/{split}/so.parquet').filter(pl.col('rid').is_in(d['ro'].unique().implode()))
                  .select(ro='rid', ao='ad', _n=strip('business_name')).collect().with_columns(no=clean('_n')).drop('_n'))
            x = d.select('r1', 'ro').join(s1, on='r1', how='left').join(so, on='ro', how='left'); del so
            parts = []
            for i in range(0, x.height, 250_000):
                y = x.slice(i, 250_000)
                args = list(zip(y['n1'].to_list(), y['no'].to_list(), y['a1'].to_list(), y['ao'].to_list()))
                try:
                    res = pool.map_async(row, args, chunksize=5000).get(timeout=600)
                except Exception as e:  # a worker died (e.g. out of memory): restart the pool and retry once
                    print('pool restart:', type(e).__name__, flush=True); pool.terminate(); pool = Pool(8)
                    res = pool.map_async(row, args, chunksize=5000).get(timeout=600)
                parts.append(np.array(res, dtype=np.float32)); del args, res
            arr = np.concatenate(parts); del parts, x
            d = d.with_columns([pl.Series(c, arr[:, i]) for i, c in enumerate(COLS)])
            d.write_parquet(f); print(f, round(time.time() - t), 's', flush=True); del d, arr

    pool.terminate()
