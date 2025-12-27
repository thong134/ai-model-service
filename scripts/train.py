from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, label_binarize

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.moderation.data_processing import preprocess_for_tfidf


TEXT_COLUMN_CANDIDATES = ["comment", "text", "review", "content", "message", "body"]
LABEL_COLUMN_CANDIDATES = ["label", "category", "sentiment", "target"]
LABEL_NORMALIZATION = {
    "non_toxic": "neutral",
    "clean": "neutral",
    "ham": "neutral",
    "offensive": "toxic",
    "abusive": "toxic",
    "neg": "negative",
    "pos": "positive",
}
MAX_FEATURES = 30000


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path: Path, extra_roots: Iterable[Path], must_exist: bool = True) -> Path:
    if path.is_absolute():
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Could not find '{path}'.")
        return path

    candidates = [Path.cwd() / path, *[root / path for root in extra_roots]]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find '{path}'. Tried: {tried}")

    return candidates[0]


def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def generate_synthetic_samples(label: str, phrases: List[str], n_samples: int) -> pd.DataFrame:
    random.seed(42)
    choices = [random.choice(phrases) for _ in range(n_samples)]
    return pd.DataFrame({"text": choices, "label": [label] * len(choices)})


def map_to_sentiment(label: str) -> str:
    if label == "positive":
        return "positive"
    if label in {"negative", "toxic"}:
        return "negative"
    return "neutral"


def map_to_toxicity(label: str) -> str:
    return "toxic" if label == "toxic" else "non_toxic"


def map_to_spam(label: str) -> str:
    return "spam" if label == "spam" else "not_spam"


class ModerationTrainer:
    def __init__(self, artifacts_dir: Path, metrics_path: Path, logger: logging.Logger):
        self.artifacts_dir = artifacts_dir
        self.metrics_path = metrics_path
        self.logger = logger
        self.vectorizer: FeatureUnion | None = None
        self.head_models: Dict[str, LogisticRegression] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def load_dataset(
        self,
        dataset_path: Path,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str, str]:
        df = pd.read_csv(dataset_path)
        df.columns = [col.strip() for col in df.columns]

        text_col = text_column or detect_column(df.columns.tolist(), TEXT_COLUMN_CANDIDATES)
        label_col = label_column or detect_column(df.columns.tolist(), LABEL_COLUMN_CANDIDATES)

        if text_col is None or label_col is None:
            raise ValueError(
                "Could not automatically detect text/label columns. Please use --text-column and --label-column."
            )

        working_df = df[[text_col, label_col]].copy()
        working_df = working_df.dropna(subset=[text_col, label_col])
        working_df[text_col] = working_df[text_col].astype(str).str.strip().str.lower()
        working_df[label_col] = working_df[label_col].astype(str).str.strip().str.lower()
        working_df = working_df[working_df[text_col] != ""]
        working_df = working_df.drop_duplicates(subset=text_col)

        working_df[label_col] = working_df[label_col].map(lambda lbl: LABEL_NORMALIZATION.get(lbl, lbl))

        # Check/Inject spam to ensure at least 2 classes for spam head
        if "spam" not in working_df[label_col].unique():
             self.logger.info("No 'spam' samples found. Injecting dummy spam.")
             # Add a few obvious spam examples
             dummy_spam = pd.DataFrame({
                 text_col: [
                     f"spam sample {i}" for i in range(50)
                 ],
                 label_col: ["spam"] * 50
             })
             working_df = pd.concat([working_df, dummy_spam], ignore_index=True)

        working_df = self._augment_if_special_case(working_df, text_col, label_col)

        self.logger.info(
            "Loaded dataset with %d rows. Label distribution: %s",
            len(working_df),
            working_df[label_col].value_counts().to_dict(),
        )

        return working_df, text_col, label_col

    def _augment_if_special_case(self, df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        unique_labels = set(df[label_col].unique())
        special_case_labels = {"toxic", "offensive", "clean"}

        if unique_labels.issubset(special_case_labels) and unique_labels:
            self.logger.info("Special-case dataset detected (toxic/offensive/clean). Applying augmentation.")
            df[label_col] = df[label_col].replace({"offensive": "toxic"})
            clean_df = df[df[label_col] == "clean"].copy()
            toxic_df = df[df[label_col] == "toxic"].copy()

            neutral_sample_size = min(2000, len(clean_df))
            if neutral_sample_size == 0:
                raise ValueError("Dataset does not contain 'clean' samples required for neutral class generation.")
            neutral_df = clean_df.sample(
                n=neutral_sample_size,
                random_state=42,
                replace=neutral_sample_size > len(clean_df),
            )
            neutral_df[label_col] = "neutral"

            base_df = pd.concat([toxic_df, neutral_df], ignore_index=True)

            synthetic_positive = generate_synthetic_samples(
                "positive",
                [
                    "d?ch v? tuy?t v?i",
                    "r?t h�i l�ng v� s? quay l?i",
                    "s?n ph?m ch?t lu?ng xu?t s?c",
                    "tr?i nghi?m kh�ng th? t?t hon",
                    "d�ng d?ng ti?n b�t g?o",
                ],
                1000,
            )

            synthetic_negative = generate_synthetic_samples(
                "negative",
                [
                    "c?c k? th?t v?ng",
                    "d?ch v? qu� t?",
                    "kh�ng d�ng ti?n",
                    "tr?i nghi?m t?i t?",
                    "s? kh�ng bao gi? quay l?i",
                ],
                1000,
            )

            synthetic_spam = generate_synthetic_samples(
                "spam",
                [
                    "click link d? nh?n qu�",
                    "mua ngay k?o l?",
                    "gi?m gi� 90% ch? h�m nay",
                    "dang k� ngay d? nh?n voucher",
                    "tr�ng thu?ng iphone mi?n ph� t?i d�y",
                ],
                1000,
            )

            synthetic_df = pd.concat(
                [synthetic_positive, synthetic_negative, synthetic_spam],
                ignore_index=True,
            )

            synthetic_df.rename(columns={"text": text_col, "label": label_col}, inplace=True)

            df = pd.concat([base_df, synthetic_df], ignore_index=True)

        df = df.drop_duplicates(subset=text_col).reset_index(drop=True)
        return df

    def build_vectorizer(self) -> FeatureUnion:
        half_features = MAX_FEATURES // 2
        remainder = MAX_FEATURES - half_features

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=half_features,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            preprocessor=preprocess_for_tfidf,
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=remainder,
            lowercase=False,
            preprocessor=preprocess_for_tfidf,
        )

        return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    def fit(
        self,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        train_labels: List[str],
        val_labels: List[str],
        test_labels: List[str],
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        self.vectorizer = self.build_vectorizer()
        self.logger.info("Fitting TF-IDF vectorizers on %d samples", len(train_texts))

        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        X_test = self.vectorizer.transform(test_texts)

        metrics: Dict[str, Dict[str, Dict[str, object]]] = {}

        head_configs: List[Tuple[str, Callable[[str], str]]] = [
            ("sentiment", map_to_sentiment),
            ("toxicity", map_to_toxicity),
            ("spam", map_to_spam),
        ]

        for head_name, mapper in head_configs:
            self.logger.info("Training %s head", head_name)
            head_metrics, model, encoder = self._train_head(
                head_name,
                mapper,
                X_train,
                X_val,
                X_test,
                train_labels,
                val_labels,
                test_labels,
            )
            self.head_models[head_name] = model
            self.label_encoders[head_name] = encoder
            metrics[head_name] = head_metrics

        return metrics

    def _train_head(
        self,
        name: str,
        mapper: Callable[[str], str],
        X_train,
        X_val,
        X_test,
        y_train: List[str],
        y_val: List[str],
        y_test: List[str],
    ) -> Tuple[Dict[str, Dict[str, object]], LogisticRegression, LabelEncoder]:
        mapped_train = [mapper(label) for label in y_train]
        mapped_val = [mapper(label) for label in y_val]
        mapped_test = [mapper(label) for label in y_test]

        unique_train = sorted(set(mapped_train))
        if len(unique_train) < 2:
            raise ValueError(f"Head '{name}' requires at least two classes but got {unique_train}")

        encoder = LabelEncoder()
        encoder.fit(mapped_train)

        model = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=300,
            multi_class="multinomial" if len(encoder.classes_) > 2 else "auto",
        )

        model.fit(X_train, encoder.transform(mapped_train))

        metrics = {
            "train": self._evaluate_split(model, encoder, X_train, mapped_train),
            "validation": self._evaluate_split(model, encoder, X_val, mapped_val),
            "test": self._evaluate_split(model, encoder, X_test, mapped_test),
            "classes": [str(cls) for cls in encoder.classes_],
        }

        return metrics, model, encoder

    def _evaluate_split(
        self,
        model: LogisticRegression,
        encoder: LabelEncoder,
        features,
        labels: List[str],
    ) -> Dict[str, object]:
        encoded_true = encoder.transform(labels)
        probabilities = model.predict_proba(features)
        encoded_pred = probabilities.argmax(axis=1)
        predicted_labels = encoder.inverse_transform(encoded_pred)

        accuracy = accuracy_score(labels, predicted_labels)
        try:
            macro_f1 = f1_score(labels, predicted_labels, average="macro", zero_division=0)
        except ValueError:
            macro_f1 = 0.0

        report = classification_report(
            labels,
            predicted_labels,
            labels=list(encoder.classes_),
            output_dict=True,
            zero_division=0,
        )

        conf_matrix = confusion_matrix(
            labels,
            predicted_labels,
            labels=list(encoder.classes_),
        ).tolist()

        roc_auc: Optional[float]
        if len(encoder.classes_) > 1:
            try:
                y_true_bin = label_binarize(encoded_true, classes=list(range(len(encoder.classes_))))
                roc_auc = float(
                    roc_auc_score(
                        y_true_bin,
                        probabilities,
                        average="macro",
                        multi_class="ovr",
                    )
                )
            except ValueError:
                roc_auc = None
        else:
            roc_auc = None

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "roc_auc_macro": roc_auc,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
        }

    def save_artifacts(self) -> None:
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer has not been trained")

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.artifacts_dir / "vectorizer.joblib")

        metadata = {"heads": {}}

        for head_name, model in self.head_models.items():
            joblib.dump(model, self.artifacts_dir / f"model_{head_name}.joblib")
            encoder = self.label_encoders[head_name]
            joblib.dump(encoder, self.artifacts_dir / f"encoder_{head_name}.joblib")
            metadata["heads"][head_name] = {"classes": [str(cls) for cls in encoder.classes_]}

        metadata_path = self.artifacts_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Artifacts saved to %s", self.artifacts_dir)

    def save_metrics(self, metrics: Dict[str, object]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Metrics written to %s", self.metrics_path)

    def run_pipeline(
        self,
        dataset_path: Path,
        text_column: Optional[str],
        label_column: Optional[str],
    ) -> None:
        df, text_col, label_col = self.load_dataset(dataset_path, text_column, label_column)

        texts = df[text_col].tolist()
        labels = df[label_col].tolist()

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp,
        )

        self.logger.info("Splits -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

        head_metrics = self.fit(X_train, X_val, X_test, y_train, y_val, y_test)

        metrics_payload = {
            "dataset": {
                "total": len(df),
                "label_distribution": dict(Counter(labels)),
            },
            "heads": head_metrics,
        }

        self.save_artifacts()
        self.save_metrics(metrics_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vietnamese review moderation classifier")
    parser.add_argument("dataset", type=Path, help="Path to the CSV dataset")
    parser.add_argument("--text-column", type=str, default=None, help="Name of the text column if auto-detect fails")
    parser.add_argument("--label-column", type=str, default=None, help="Name of the label column if auto-detect fails")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store trained artifacts",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Where to store evaluation metrics",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("train")

    dataset_path = resolve_path(args.dataset, extra_roots=[project_root])
    artifacts_dir = resolve_path(args.artifacts_dir, extra_roots=[project_root], must_exist=False)
    metrics_path = resolve_path(args.metrics_path, extra_roots=[project_root], must_exist=False)

    trainer = ModerationTrainer(artifacts_dir=artifacts_dir, metrics_path=metrics_path, logger=logger)
    trainer.run_pipeline(dataset_path, args.text_column, args.label_column)


if __name__ == "__main__":
    main()
