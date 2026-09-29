"""
Candidate generation / blocking.

Strategy (see experiments/data_audit_report.md for why): country match is
mandatory (0% mismatch measured on true pairs), so every index is partitioned
by normalized country first. Within a country we UNION five independent
blocking channels, because the audit showed no single one is sufficient:

  1. exact normalized name
  2. exact normalized address
  3. name-token overlap (inverted index on individual name tokens)
  4. address numeric-token overlap (street numbers / PIN-like fragments —
     the most reliable anchor when names are transliterated/garbled/missing)
  5. address character n-gram overlap (recovers typo-level address variants
     with zero exact token overlap, e.g. "Wanye" vs "Wayne")

Any token whose posting list is too long (a generic/very common token) is
skipped for blocking purposes to avoid candidate-set explosion — it would
contribute little discriminative signal anyway.

The index is a set of plain dict[str, list[str]] structures kept in memory,
partitioned by country. This is the memory-heavy part of the pipeline: for
the full ~10M S2+S3 rows this needs a few GB of RAM. On this sandbox we run
it over a bounded subset (see pipeline.py); the same code operates unchanged
on the full files given more memory/time.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from . import config
from .normalize import normalize_name, normalize_address, normalize_country, char_ngrams


@dataclass
class Record:
    entity_id: str
    country: str
    name_norm: str
    name_tokens: Tuple[str, ...]
    addr_norm: str
    addr_numeric: Tuple[str, ...]
    addr_ngrams: Tuple[str, ...]


def build_record(entity_id: str, raw_name: str, raw_addr: str, raw_country: str,
                  ngram_n: int = 4) -> Record:
    nf = normalize_name(raw_name)
    af = normalize_address(raw_addr)
    country = normalize_country(raw_country)
    grams = char_ngrams(af.normalized, n=ngram_n) | char_ngrams(nf.normalized, n=ngram_n)
    return Record(
        entity_id=entity_id,
        country=country,
        name_norm=nf.normalized,
        name_tokens=tuple(nf.tokens),
        addr_norm=af.normalized,
        addr_numeric=tuple(sorted(af.numeric_tokens or ())),
        addr_ngrams=tuple(grams),
    )


class CountryIndex:
    """All blocking indexes for a single normalized-country partition."""
    def __init__(self):
        self.exact_name: Dict[str, List[str]] = defaultdict(list)
        self.exact_addr: Dict[str, List[str]] = defaultdict(list)
        self.name_token: Dict[str, List[str]] = defaultdict(list)
        self.addr_numeric: Dict[str, List[str]] = defaultdict(list)
        self.addr_ngram: Dict[str, List[str]] = defaultdict(list)
        self.n_records: int = 0

    def add(self, rec: Record, cfg: config.BlockingConfig):
        self.n_records += 1
        if cfg.use_exact_name_block and rec.name_norm:
            self.exact_name[rec.name_norm].append(rec.entity_id)
        if cfg.use_exact_address_block and rec.addr_norm:
            self.exact_addr[rec.addr_norm].append(rec.entity_id)
        if cfg.use_name_token_block:
            for t in rec.name_tokens:
                if len(t) >= cfg.min_name_token_len:
                    self.name_token[t].append(rec.entity_id)
        if cfg.use_numeric_token_block:
            for t in rec.addr_numeric:
                if len(t) >= cfg.min_numeric_token_len:
                    self.addr_numeric[t].append(rec.entity_id)
        if cfg.use_address_ngram_block:
            for g in rec.addr_ngrams:
                self.addr_ngram[g].append(rec.entity_id)


class BlockingIndex:
    """Country-partitioned union of S2 + S3 blocking indexes."""
    def __init__(self, cfg: config.BlockingConfig = config.BLOCKING):
        self.cfg = cfg
        self.by_country: Dict[str, CountryIndex] = defaultdict(CountryIndex)
        self._n_added = 0

    def add_record(self, entity_id: str, name: str, address: str, country: str):
        rec = build_record(entity_id, name, address, country, ngram_n=self.cfg.address_ngram_n)
        self.by_country[rec.country].add(rec, self.cfg)
        self._n_added += 1

    def candidates_for(self, name: str, address: str, country: str) -> Set[str]:
        """Union every blocking channel, but when the union exceeds
        max_candidates_per_s1 we must not truncate arbitrarily (e.g. sorted-
        alphabetical truncation silently drops true matches at random and
        collapses blocking recall). Instead we accumulate a weighted "vote"
        per candidate — how many independent channels, and how strong a
        channel, flagged it — and keep the highest-voted candidates. A
        candidate hit by an exact-match channel or a numeric address token
        (both highly discriminative, per the audit) always outranks one hit
        only by a couple of generic character n-grams."""
        rec = build_record("__query__", name, address, country, ngram_n=self.cfg.address_ngram_n)
        idx = self.by_country.get(rec.country)
        if idx is None:
            return set()
        cfg = self.cfg
        votes: Dict[str, float] = defaultdict(float)

        def _add_votes(index_dict, keys, cap, weight):
            for k in keys:
                postings = index_dict.get(k)
                if postings and len(postings) <= cap:
                    for pid in postings:
                        votes[pid] += weight

        if cfg.use_exact_name_block and rec.name_norm:
            for pid in idx.exact_name.get(rec.name_norm, []):
                votes[pid] += 10.0
        if cfg.use_exact_address_block and rec.addr_norm:
            for pid in idx.exact_addr.get(rec.addr_norm, []):
                votes[pid] += 10.0
        if cfg.use_numeric_token_block:
            keys = [t for t in rec.addr_numeric if len(t) >= cfg.min_numeric_token_len]
            _add_votes(idx.addr_numeric, keys, cfg.max_postings_per_token, weight=3.0)
        if cfg.use_name_token_block:
            keys = [t for t in rec.name_tokens if len(t) >= cfg.min_name_token_len]
            _add_votes(idx.name_token, keys, cfg.max_name_postings_per_token, weight=1.5)
        if cfg.use_address_ngram_block:
            _add_votes(idx.addr_ngram, rec.addr_ngrams, cfg.max_ngram_postings_per_token, weight=0.15)

        if not votes:
            return set()
        if len(votes) <= cfg.max_candidates_per_s1:
            return set(votes.keys())
        # Keep the highest-voted candidates; break ties deterministically by id.
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        return {pid for pid, _ in ranked[: cfg.max_candidates_per_s1]}

    def stats(self) -> dict:
        return {
            "n_records_indexed": self._n_added,
            "n_countries": len(self.by_country),
            "records_per_country": {c: idx.n_records for c, idx in self.by_country.items()},
        }
