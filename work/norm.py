import polars as pl

NAME_STOP = ["inc","llc","ltd","limited","private","pvt","corp","corporation","co","company","the","and",
 "of","services","service","center","centre","group","partners","www","com","net","org","in","dba","ta",
 "incorporated","llp","pc","plc","sa","sarl","sas","eurl","holdings","l","p","d","b","a","t","s","n","et","de"]
ADDR_STOP = ["street","st","road","rd","avenue","ave","unit","apt","apartment","po","box","suite","ste","floor",
 "near","opp","no","drive","dr","null","lane","ln","court","ct","place","pl","boulevard","blvd","circle","cir",
 "way","h","hno","house","building","bldg","flat","rue","de","du","la","le","des","of","city","town","township",
 "the","and","plot","east","west","north","south","e","w","n","s","trail","terrace","ter","hwy","highway",
 "pkwy","parkway","village","borough","twp","none","nd","rd","th","1st"]

def clean(col):
    return (pl.col(col).fill_null("").str.to_lowercase().str.normalize("NFKD")
            .str.replace_all(r"\p{M}", "").str.replace_all(r"<null>|\bnull\b|\bnone\b", " ")
            .str.replace_all(r"(\d+)(st|nd|rd|th)\b", "$1")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.replace_all(r"\b0+(\d)", "$1").str.strip_chars())

def load(path):
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy")
    df = df.with_columns(nm=clean("business_name"), ad=clean("business_address"),
                           country=pl.col("country").str.to_lowercase())
    df = df.with_columns(sk=pl.col("business_name").fill_null("").map_elements(skel, return_dtype=pl.String),
                         core=pl.col("nm").map_elements(core_name, return_dtype=pl.String))
    codes = (pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"<null>|null", " ")
             .str.split(",").list.eval(pl.element().str.strip_chars().str.split(" ")).list.eval(pl.element().flatten())
             .list.eval(pl.element().str.replace_all(r"^(?:h\.?no|d\.?no|no|plot|flat|shop|door|room|unit|#)[\.:]*", "")
                        .str.replace_all(r"[^a-z0-9]", "").str.replace_all(r"^0+", ""))
             .list.eval(pl.element().filter(pl.element().str.contains(r"\d") & (pl.element().str.len_chars() >= 3))).list.unique())
    return df.with_columns(codes=codes)

def tokens(df):
    """(rid, tok:u64) blocking keys: name/addr unigrams, concat-name, and order-free token pairs."""
    b = df.select("rid",
        n=pl.col("nm").str.split(" ").list.eval(pl.element().filter(
            (pl.element().str.len_chars() >= 2) & ~pl.element().is_in(NAME_STOP))).list.unique(maintain_order=True).list.head(4),
        a=pl.col("ad").str.split(" ").list.eval(pl.element().filter(
            (pl.element().str.len_chars() >= 1) & ~pl.element().is_in(ADDR_STOP))).list.unique(maintain_order=True).list.head(8),
        c=pl.col("nm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(NAME_STOP))).list.join(""),
        k=pl.col("sk").str.split(" ").list.unique(maintain_order=True).list.head(4), kc=pl.col("sk").str.replace_all(" ", ""))
    n = b.select("rid", t=pl.col("n")).explode("t").drop_nulls().with_columns(h=pl.col("t").hash(1), i=pl.int_range(pl.len()).over("rid"))
    a = b.select("rid", t=pl.col("a")).explode("t").drop_nulls().with_columns(h=pl.col("t").hash(2), i=pl.int_range(pl.len()).over("rid"))
    # single-char/number-only unigrams are useless alone; keep them only inside pairs
    uni = pl.concat([n.filter(pl.col("t").str.len_chars() >= 3).select("rid", tok="h"),
                     a.filter(pl.col("t").str.len_chars() >= 3, ~pl.col("t").str.contains(r"^\d+$")).select("rid", tok="h"),
                     b.filter(pl.col("c").str.len_chars() >= 4).select("rid", tok=pl.col("c").hash(3))])
    k = b.select("rid", t=pl.col("k")).explode("t").drop_nulls().filter(pl.col("t").str.len_chars() >= 2).with_columns(h=pl.col("t").hash(5), i=pl.int_range(pl.len()).over("rid"))
    kk = k.join(k, on="rid").filter(pl.col("i") < pl.col("i_right")).select("rid", tok=(pl.col("h") ^ pl.col("h_right")).hash(8))
    ka = k.filter(pl.col("i") < 3).join(a, on="rid").select("rid", tok=(pl.col("h") ^ pl.col("h_right")).hash(6))
    uni = pl.concat([uni, k.filter(pl.col("t").str.len_chars() >= 3).select("rid", tok="h"),
                     b.filter(pl.col("kc").str.len_chars() >= 4).select("rid", tok=pl.col("kc").hash(7))])
    cd = df.select("rid", t=pl.col("codes")).explode("t").drop_nulls().with_columns(h=pl.col("t").hash(10))
    uni = pl.concat([uni, cd.filter(pl.col("t").str.len_chars() >= 4).select("rid", tok="h")])
    kcd = k.filter(pl.col("i") < 2).join(cd, on="rid").select("rid", tok=(pl.col("h") ^ pl.col("h_right")).hash(11))
    aa = a.join(a, on="rid").filter(pl.col("i") < pl.col("i_right")).select("rid", tok=pl.col("h") ^ pl.col("h_right"))
    nn = n.join(n, on="rid").filter(pl.col("i") < pl.col("i_right")).select("rid", tok=pl.col("h") ^ pl.col("h_right"))
    na = n.filter(pl.col("i") < 3).join(a, on="rid").select("rid", tok=(pl.col("h") ^ pl.col("h_right")).hash(4))
    return pl.concat([uni, aa, nn, na, kk, ka, kcd]).unique()

# ---- phonetic skeleton (handles Indic-script names via transliteration, and Latin typos) ----
import re as _re
from unidecode import unidecode as _ud
_MAP = [('ph','f'),('th','t'),('sh','s'),('ch','c'),('kh','k'),('gh','g'),('dh','d'),('bh','b'),('w','v'),
        ('q','k'),('ck','k'),('x','ks'),('z','j'),('y','i'),('c','k'),('b','v'),('g','j'),('d','t')]
SK_STOP = {'prvt','lmt','lp','llp','ink','llk','ltt','pvt','kmpn','krp','krprtn','srvks','srvk','sntr','jrp','intrprs'}
def skel(s):
    if not s: return ''
    s = _ud(s).lower() if not s.isascii() else s.lower()
    s = _re.sub(r'[^a-z ]', ' ', s)
    for a, b in _MAP: s = s.replace(a, b)
    out = []
    for w in s.split():
        k = _re.sub(r'(.)\1+', r'\1', w[0] + _re.sub('[aeiou]', '', w[1:]))
        if len(k) >= 2 and k not in SK_STOP: out.append(k)
    return ' '.join(out)

# ---- core name (filler words removed, OCR digit->letter fixed, order-free) for the no-address name key ----
_GEN = set("""com ltd center centre services service inc llc co corporation corp the partners lp limited smt incorporated mr mrs ms
shri sri dr dba formerly as www company llp and trading doing business aka nee enterprises fka known labs sys one pc trust board
foundation society association authority district commission council federation associates associate group pvt private pllc
india care clinic solutions of brothers technologies technology global industries holdings international ventures m s in id st""".split())
_OCR = str.maketrans('0158634', 'olsbgea')
def core_name(s):
    out = set()
    for t in (s or '').split():
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t): t = t.translate(_OCR)
        if len(t) > 1 and t not in _GEN: out.add(t)
    return ' '.join(sorted(out))
