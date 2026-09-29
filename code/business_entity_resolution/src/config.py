"""
Central configuration for the Business Entity Resolution pipeline.

All paths are relative to the directory the pipeline is *run from*
(the `student_resource/` / competition root), so nothing here hard-codes
an absolute or OS-specific path. Override any of these with CLI flags
exposed by pipeline.py where relevant.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Paths:
    train_source1: str = "dataset/train/train_source1.tsv"
    train_source2: str = "dataset/train/train_source2.tsv"
    train_source3: str = "dataset/train/train_source3.tsv"
    train_ground_truth: str = "dataset/train/train_ground_truth.tsv"

    test_source1: str = "dataset/test/test_source1.tsv"
    test_source2: str = "dataset/test/test_source2.tsv"
    test_source3: str = "dataset/test/test_source3.tsv"

    output_dir: str = "output"
    matching_results: str = "output/matching_results.tsv"
    candidate_pairs: str = "output/candidate_pairs.tsv"

    models_dir: str = "models/saved_models"
    experiments_csv: str = "experiments/experiments.csv"


@dataclass
class BlockingConfig:
    # Address numeric-token blocking: entities sharing a rare-enough numeric
    # token (street number, PIN/ZIP-like fragment) in the same country are candidates.
    use_numeric_token_block: bool = True
    # Skip numeric tokens that are too common (would create a giant candidate list),
    # e.g. purely single-digit tokens like "1".
    min_numeric_token_len: int = 2
    max_postings_per_token: int = 400  # drop tokens that block-explode (stopword-like numbers)

    # Name-token blocking: entities sharing a rare name token in the same country.
    use_name_token_block: bool = True
    min_name_token_len: int = 3
    max_name_postings_per_token: int = 400

    # Address character n-gram blocking (helps recover typo-level address variants
    # even when there is zero exact token overlap, e.g. "Wanye" vs "Wayne").
    use_address_ngram_block: bool = True
    address_ngram_n: int = 4
    max_ngram_postings_per_token: int = 300

    # Exact normalized name / address blocking (cheap, high precision, catches easy cases).
    use_exact_name_block: bool = True
    use_exact_address_block: bool = True

    # Hard cap on candidates kept per S1 after unioning all strategies (safety valve
    # against pathological blocking explosion on generic names/addresses).
    max_candidates_per_s1: int = 100


@dataclass
class ModelConfig:
    model_type: str = "lightgbm"  # one of: logreg, xgboost, lightgbm
    random_state: int = 42
    # LightGBM / XGBoost params tuned for a small, memory-constrained box.
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    negative_per_positive: int = 6  # ratio of (hard+random) negatives sampled per positive pair
    hard_negative_fraction: float = 0.6  # fraction of sampled negatives that are "hard"


@dataclass
class ThresholdConfig:
    candidates: List[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    default: float = 0.75


PATHS = Paths()
BLOCKING = BlockingConfig()
MODEL = ModelConfig()
THRESHOLDS = ThresholdConfig()

RANDOM_STATE = 42
CHUNK_SIZE = 25_000  # rows per chunk when streaming large TSVs
