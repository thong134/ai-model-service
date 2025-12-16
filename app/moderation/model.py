from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from .config import AppConfig


@dataclass
class HeadPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class ModerationPrediction:
    sentiment: HeadPrediction
    toxicity: HeadPrediction
    spam: HeadPrediction


@dataclass
class ModelHead:
    name: str
    model_path: Path
    encoder_path: Path
    model: BaseEstimator | None = None
    encoder: LabelEncoder | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing '{self.name}' model artifact at {self.model_path}")
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Missing '{self.name}' encoder artifact at {self.encoder_path}")

        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"Model head '{self.name}' is not loaded")

        probabilities = self.model.predict_proba(features)
        return np.asarray(probabilities, dtype=float)

    def build_prediction(self, proba_row: np.ndarray) -> HeadPrediction:
        if self.encoder is None:
            raise RuntimeError(f"Encoder for head '{self.name}' is not loaded")

        labels = [str(label) for label in self.encoder.classes_]
        scores = {label: float(prob) for label, prob in zip(labels, proba_row)}

        top_label, top_score = max(scores.items(), key=lambda item: item[1])
        return HeadPrediction(label=top_label, confidence=float(top_score), probabilities=scores)


class ModerationModel:
    def __init__(self, config: AppConfig):
        self.config = config
        self.vectorizer: Any = None
        paths = self.config.paths
        self.heads: Dict[str, ModelHead] = {
            "sentiment": ModelHead("sentiment", paths.sentiment_model_path, paths.sentiment_encoder_path),
            "toxicity": ModelHead("toxicity", paths.toxicity_model_path, paths.toxicity_encoder_path),
            "spam": ModelHead("spam", paths.spam_model_path, paths.spam_encoder_path),
        }
        self.metadata: Dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        vectorizer_path = self.config.paths.vectorizer_path
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Missing vectorizer artifact at {vectorizer_path}")

        self.vectorizer = joblib.load(vectorizer_path)
        for head in self.heads.values():
            head.load()

        self.metadata = self._load_metadata()
        self._loaded = True

    def _load_metadata(self) -> Dict[str, Any]:
        metadata_path = self.config.paths.metadata_path
        if metadata_path.exists():
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model artifacts have not been loaded. Call load() first.")

    def predict(self, texts: Iterable[str]) -> List[ModerationPrediction]:
        self._ensure_loaded()
        inputs = [text for text in texts]
        if not inputs:
            return []

        features = self.vectorizer.transform(inputs)
        head_outputs: Dict[str, List[HeadPrediction]] = {}

        for name, head in self.heads.items():
            probabilities = head.predict_proba(features)
            head_outputs[name] = [head.build_prediction(row) for row in probabilities]

        predictions: List[ModerationPrediction] = []
        for idx in range(len(inputs)):
            predictions.append(
                ModerationPrediction(
                    sentiment=head_outputs["sentiment"][idx],
                    toxicity=head_outputs["toxicity"][idx],
                    spam=head_outputs["spam"][idx],
                )
            )

        return predictions

    def predict_one(self, text: str) -> ModerationPrediction:
        predictions = self.predict([text])
        if not predictions:
            raise RuntimeError("Prediction failed: no outputs returned")
        return predictions[0]

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        predictions = self.predict(texts)
        toxic_scores = [self.probability_for_label(pred.toxicity, "toxic") for pred in predictions]
        return np.asarray(toxic_scores, dtype=float)

    @staticmethod
    def probability_for_label(head_prediction: HeadPrediction, label_name: str) -> float:
        target = label_name.lower()
        for label, score in head_prediction.probabilities.items():
            if label.lower() == target:
                return float(score)

        for label, score in head_prediction.probabilities.items():
            if target in label.lower():
                return float(score)

        if len(head_prediction.probabilities) == 2:
            values = list(head_prediction.probabilities.values())
            if head_prediction.label.lower() == target:
                return float(max(values))
            return float(min(values))

        return 0.0


__all__ = [
    "HeadPrediction",
    "ModerationModel",
    "ModerationPrediction",
    "ModelHead",
]
