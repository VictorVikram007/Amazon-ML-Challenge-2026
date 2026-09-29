"""
Normalization for business_name / business_address / country.

Design goals (see experiments/data_audit_report.md for the real examples that
motivated each rule):
- Conservative: never destroy the original string; always keep it alongside
  the normalized version.
- Handle real noise observed in the data: literal "null"/"<NULL>" tokens,
  legal-suffix variants (Inc/Incorporated, LLC/L.L.C., Pvt/Private, Ltd/Limited,
  Corp/Corporation), "&" vs "and", punctuation/whitespace noise, mixed case.
  Non-Latin scripts (Devanagari/Tamil/Telugu/Kannada, seen in the audit) are
  NOT transliterated (no external translation allowed) — they are Unicode-
  normalized and tokenized as-is so exact/token matching still works when a
  same-script duplicate exists, but we do not pretend to bridge script boundaries.
- Address: extract numeric tokens (street numbers, PIN/ZIP-like fragments)
  separately, since audit showed these are the most reliable anchor when
  names are corrupted, transliterated, or missing.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Set

# Noise tokens observed literally inside address strings (not NaN, but junk).
_NULL_TOKENS = {"null", "<null>", "n/a", "na", "none", "nil"}

# Legal-suffix / common abbreviation normalization map (applied on whole tokens,
# after punctuation stripping). Conservative: only well-attested equivalences.
_LEGAL_SUFFIX_MAP = {
    "incorporated": "inc", "inc.": "inc",
    "corporation": "corp", "corp.": "corp",
    "limited": "ltd", "ltd.": "ltd",
    "private": "pvt", "pvt.": "pvt",
    "company": "co", "co.": "co",
    "llp.": "llp", "llc.": "llc",
    "l.l.c": "llc", "l.l.p": "llp",
}

_ADDRESS_ABBR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "terrace": "ter",
    "place": "pl", "apartment": "apt", "building": "bldg",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def _strip_null_tokens(text: str) -> str:
    parts = re.split(r"[,\|;]", text)
    kept = [p for p in parts if p.strip().lower() not in _NULL_TOKENS]
    return ",".join(kept)


def _basic_clean(text: Optional[str]) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _strip_null_tokens(text)
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def normalize_name(raw: Optional[str]) -> "NormalizedField":
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    norm_tokens = [_LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    # Drop very common trailing "filler" legal-entity tokens for a *secondary*
    # "core name" representation used only for a stricter exact-match block —
    # we keep the full token list too, so nothing is destroyed.
    normalized = " ".join(norm_tokens)
    return NormalizedField(original=raw or "", normalized=normalized, tokens=norm_tokens)


def normalize_address(raw: Optional[str]) -> "NormalizedField":
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    norm_tokens = [_ADDRESS_ABBR_MAP.get(t, t) for t in tokens]
    normalized = " ".join(norm_tokens)
    numeric_tokens = set(_DIGIT_RUN_RE.findall(normalized))
    return NormalizedField(
        original=raw or "", normalized=normalized, tokens=norm_tokens,
        numeric_tokens=numeric_tokens,
    )


def normalize_country(raw: Optional[str]) -> str:
    """Countries are an open, unbounded string set (train has US/India, test
    adds France, more could appear). We only fold case/whitespace — never a
    hard-coded lookup table — so unseen values pass through untouched."""
    if raw is None:
        return ""
    return _WS_RE.sub(" ", str(raw).strip().lower())


@dataclass
class NormalizedField:
    original: str
    normalized: str
    tokens: List[str]
    numeric_tokens: Optional[Set[str]] = None

    def token_set(self) -> Set[str]:
        return set(self.tokens)


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Character n-grams over the normalized (whitespace-joined) string, used
    for typo-tolerant blocking/similarity (e.g. 'wanye' vs 'wayne' share grams)."""
    padded = text.replace(" ", "")
    if len(padded) < n:
        return {padded} if padded else set()
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}
