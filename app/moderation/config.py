from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PathConfig:
    lang: str = "vi"
    base_dir: Path = Path("artifacts")
    
    @property
    def lang_dir(self) -> Path:
        return self.base_dir / self.lang

    @property
    def vectorizer_path(self) -> Path: return self.lang_dir / "vectorizer.joblib"
    @property
    def sentiment_model_path(self) -> Path: return self.lang_dir / "model_sentiment.joblib"
    @property
    def sentiment_encoder_path(self) -> Path: return self.lang_dir / "encoder_sentiment.joblib"
    @property
    def toxicity_model_path(self) -> Path: return self.lang_dir / "model_toxicity.joblib"
    @property
    def toxicity_encoder_path(self) -> Path: return self.lang_dir / "encoder_toxicity.joblib"
    @property
    def spam_model_path(self) -> Path: return self.lang_dir / "model_spam.joblib"
    @property
    def spam_encoder_path(self) -> Path: return self.lang_dir / "encoder_spam.joblib"
    @property
    def metadata_path(self) -> Path: return self.lang_dir / "metadata.json"


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

    @classmethod
    def for_lang(cls, lang: str) -> AppConfig:
        return cls(paths=PathConfig(lang=lang))
