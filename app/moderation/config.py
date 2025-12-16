from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PathConfig:
    base_dir: Path = Path("artifacts")
    vectorizer_path: Path = base_dir / "vectorizer.joblib"
    sentiment_model_path: Path = base_dir / "model_sentiment.joblib"
    sentiment_encoder_path: Path = base_dir / "encoder_sentiment.joblib"
    toxicity_model_path: Path = base_dir / "model_toxicity.joblib"
    toxicity_encoder_path: Path = base_dir / "encoder_toxicity.joblib"
    spam_model_path: Path = base_dir / "model_spam.joblib"
    spam_encoder_path: Path = base_dir / "encoder_spam.joblib"
    metadata_path: Path = base_dir / "metadata.json"


@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    c: float = 4.0
    max_iter: int = 300
    class_weight: str = "balanced"
    solver: str = "lbfgs"


@dataclass
class InferenceConfig:
    toxicity_manual_threshold: float = 0.4
    toxicity_reject_threshold: float = 0.7
    spam_manual_threshold: float = 0.4
    spam_reject_threshold: float = 0.6
    rule_manual_threshold: float = 0.4
    rule_reject_threshold: float = 0.8


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
