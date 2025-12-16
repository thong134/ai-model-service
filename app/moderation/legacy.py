from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .config import AppConfig
from .data_processing import preprocess_for_vectorizer


def _preprocess_for_model(text: str) -> str:
    return preprocess_for_vectorizer(text, strip_diacritics=False)


@dataclass
class DatasetSplits:
    train_texts: List[str]
    val_texts: List[str]
    test_texts: List[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class ReviewModel:
    def __init__(self, config: AppConfig):
        self.config = config
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: LogisticRegression | None = None
        self.label_mapping: Dict[int, str] = {0: "non_toxic", 1: "toxic"}

    def _split_dataset(self, texts: Iterable[str], labels: Iterable[int]) -> DatasetSplits:
        texts = list(texts)
        labels = np.array(list(labels))
        cfg = self.config.training

        test_and_val = cfg.test_size + cfg.validation_size
        x_train, x_temp, y_train, y_temp = train_test_split(
            texts,
            labels,
            test_size=test_and_val,
            random_state=cfg.random_state,
            stratify=labels,
        )

        relative_val = cfg.validation_size / test_and_val if test_and_val > 0 else 0.0

        if relative_val > 0:
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp,
                y_temp,
                test_size=relative_val,
                random_state=cfg.random_state,
                stratify=y_temp,
            )
        else:
            x_val, y_val = [], np.array([])
            x_test, y_test = x_temp, y_temp

        return DatasetSplits(
            train_texts=x_train,
            val_texts=x_val,
            test_texts=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> Dict[str, Any]:
        splits = self._split_dataset(texts, labels)
        cfg = self.config.training

        self.vectorizer = TfidfVectorizer(
            preprocessor=_preprocess_for_model,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
        )
        x_train_vec = self.vectorizer.fit_transform(splits.train_texts)
        x_val_vec = self.vectorizer.transform(splits.val_texts) if splits.val_texts else None
        x_test_vec = self.vectorizer.transform(splits.test_texts) if splits.test_texts else None

        self.classifier = LogisticRegression(
            C=cfg.c,
            class_weight=cfg.class_weight,
            max_iter=cfg.max_iter,
            solver=cfg.solver,
        )
        self.classifier.fit(x_train_vec, splits.y_train)

        metrics: Dict[str, Any] = {}
        metrics["train_report"] = classification_report(
            splits.y_train,
            self.classifier.predict(x_train_vec),
            output_dict=True,
            zero_division=0,
        )

        if x_val_vec is not None:
            metrics["val_report"] = classification_report(
                splits.y_val,
                self.classifier.predict(x_val_vec),
                output_dict=True,
                zero_division=0,
            )

        if x_test_vec is not None:
            metrics["test_report"] = classification_report(
                splits.y_test,
                self.classifier.predict(x_test_vec),
                output_dict=True,
                zero_division=0,
            )

        metrics["class_mapping"] = self.label_mapping
        return metrics

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("Model is not loaded")
        vectors = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(vectors)[:, 1]
        return probabilities

    def save(self) -> None:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("Model is not trained")
        paths = self.config.paths
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, paths.vectorizer_path)
        joblib.dump(self.classifier, paths.model_path)
        metadata = {"class_mapping": self.label_mapping}
        paths.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        paths = self.config.paths
        if not paths.vectorizer_path.exists() or not paths.model_path.exists():
            raise FileNotFoundError("Model artifacts not found. Please train the model first.")
        self.vectorizer = joblib.load(paths.vectorizer_path)
        self.classifier = joblib.load(paths.model_path)
        if paths.metadata_path.exists():
            metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
            mapping = metadata.get("class_mapping")
            if isinstance(mapping, dict):
                self.label_mapping = {int(k): v for k, v in mapping.items()}

    def get_label(self, proba: float, threshold: float = 0.5) -> str:
        return self.label_mapping[int(proba >= threshold)]
