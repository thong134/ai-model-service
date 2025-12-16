from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class HybridRecommenderModel:
    users: pd.DataFrame
    destinations: pd.DataFrame
    feedback: pd.DataFrame
    description_vectorizer: TfidfVectorizer
    category_encoder: OneHotEncoder
    province_encoder: OneHotEncoder
    rating_scaler: StandardScaler
    content_matrix: np.ndarray
    content_norms: np.ndarray
    user_item_matrix: pd.DataFrame
    user_factors: Optional[np.ndarray]
    collaborative_model: Optional[TruncatedSVD]
    mean_item_ratings: np.ndarray
    rating_counts: np.ndarray
    config: Dict[str, object]
    destination_index: Dict[str, int] = field(init=False)
    user_index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.destinations = self.destinations.copy()
        self.users = self.users.copy()
        self.feedback = self.feedback.copy()

        self.destinations["destinationId"] = self.destinations["destinationId"].astype(str)
        self.users["userId"] = self.users["userId"].astype(str)
        self.feedback["userId"] = self.feedback["userId"].astype(str)
        self.feedback["destinationId"] = self.feedback["destinationId"].astype(str)

        self.destination_index = {
            dest_id: idx for idx, dest_id in enumerate(self.destinations["destinationId"].tolist())
        }
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_item_matrix.index.tolist())}

    def to_payload(self) -> Dict[str, object]:
        return {
            "users": self.users,
            "destinations": self.destinations,
            "feedback": self.feedback,
            "description_vectorizer": self.description_vectorizer,
            "category_encoder": self.category_encoder,
            "province_encoder": self.province_encoder,
            "rating_scaler": self.rating_scaler,
            "content_matrix": self.content_matrix,
            "content_norms": self.content_norms,
            "user_item_matrix": self.user_item_matrix,
            "user_factors": self.user_factors,
            "collaborative_model": self.collaborative_model,
            "mean_item_ratings": self.mean_item_ratings,
            "rating_counts": self.rating_counts,
            "config": self.config,
        }


def train_model(
    users_path: str | Path = Path("data/users.csv"),
    destinations_path: str | Path = Path("data/destinations.csv"),
    feedback_path: str | Path = Path("data/feedback.csv"),
    *,
    latent_factors: int = 32,
    positive_feedback_threshold: float = 4.0,
    min_profile_items: int = 3,
    weights: Optional[Dict[str, float]] = None,
) -> HybridRecommenderModel:
    """Train a hybrid recommender that blends collaborative and content signals."""
    users_df = _load_dataframe(users_path, required_columns=["userId", "favoriteCategory", "favoriteProvince"])
    dest_df = _load_dataframe(
        destinations_path,
        required_columns=["destinationId", "category", "province", "averageRating", "description"],
    )
    feedback_df = _load_dataframe(feedback_path, required_columns=["userId", "destinationId", "rating"])

    users_df["userId"] = users_df["userId"].astype(str)
    dest_df["destinationId"] = dest_df["destinationId"].astype(str)
    feedback_df["userId"] = feedback_df["userId"].astype(str)
    feedback_df["destinationId"] = feedback_df["destinationId"].astype(str)

    dest_df = dest_df.copy()
    dest_df["description"] = dest_df["description"].fillna("")
    dest_df["category"] = dest_df["category"].fillna("unknown")
    dest_df["province"] = dest_df["province"].fillna("unknown")
    dest_df["averageRating"] = pd.to_numeric(dest_df["averageRating"], errors="coerce")
    dest_df["averageRating"] = dest_df["averageRating"].fillna(dest_df["averageRating"].mean())

    feedback_df = feedback_df.copy()
    feedback_df["rating"] = pd.to_numeric(feedback_df["rating"], errors="coerce")
    feedback_df = feedback_df.dropna(subset=["rating"])

    desc_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    desc_matrix = desc_vectorizer.fit_transform(dest_df["description"]).toarray()

    category_encoder = _make_one_hot_encoder()
    category_matrix = category_encoder.fit_transform(dest_df[["category"]])

    province_encoder = _make_one_hot_encoder()
    province_matrix = province_encoder.fit_transform(dest_df[["province"]])

    rating_scaler = StandardScaler()
    rating_matrix = rating_scaler.fit_transform(dest_df[["averageRating"]])

    content_matrix = np.hstack([desc_matrix, category_matrix, province_matrix, rating_matrix]).astype(np.float32)
    content_norms = np.linalg.norm(content_matrix, axis=1)
    content_norms = np.where(content_norms == 0.0, 1.0, content_norms)

    user_item_matrix = feedback_df.pivot_table(
        index="userId",
        columns="destinationId",
        values="rating",
        aggfunc="mean",
    ).reindex(columns=dest_df["destinationId"], fill_value=0.0)
    user_item_matrix = user_item_matrix.fillna(0.0)

    collaborative_model: Optional[TruncatedSVD] = None
    user_factors: Optional[np.ndarray] = None
    if not user_item_matrix.empty and user_item_matrix.shape[1] >= 2 and user_item_matrix.shape[0] >= 2:
        max_components = min(user_item_matrix.shape) - 1
        if max_components >= 1 and latent_factors > 0:
            n_components = max(1, min(latent_factors, max_components))
            collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = collaborative_model.fit_transform(user_item_matrix.to_numpy())
    else:
        user_item_matrix = pd.DataFrame(
            np.zeros((len(users_df["userId"].unique()), len(dest_df))),
            index=users_df["userId"].unique(),
            columns=dest_df["destinationId"],
        )

    mean_ratings = (
        feedback_df.groupby("destinationId")["rating"].mean()
        .reindex(dest_df["destinationId"])
        .fillna(dest_df["averageRating"])
        .to_numpy()
        .astype(np.float32)
    )

    rating_counts = (
        feedback_df.groupby("destinationId")
        .size()
        .reindex(dest_df["destinationId"])
        .fillna(0)
        .to_numpy()
        .astype(np.float32)
    )

    final_weights = _resolve_weights(weights)

    config = {
        "positive_feedback_threshold": float(positive_feedback_threshold),
        "min_profile_items": int(max(1, min_profile_items)),
        "weights": final_weights,
    }

    return HybridRecommenderModel(
        users=users_df,
        destinations=dest_df,
        feedback=feedback_df,
        description_vectorizer=desc_vectorizer,
        category_encoder=category_encoder,
        province_encoder=province_encoder,
        rating_scaler=rating_scaler,
        content_matrix=content_matrix,
        content_norms=content_norms,
        user_item_matrix=user_item_matrix,
        user_factors=user_factors,
        collaborative_model=collaborative_model,
        mean_item_ratings=mean_ratings,
        rating_counts=rating_counts,
        config=config,
    )


def recommend_for_user(
    model: HybridRecommenderModel,
    user_id: str,
    top_n: int = 10,
    *,
    include_rated: bool = False,
) -> List[Dict[str, object]]:
    """Return the top N destination recommendations for the given user."""
    user_id = str(user_id)
    num_items = len(model.destinations)
    if num_items == 0:
        return []

    collab_component = _collaborative_scores(model, user_id)
    content_component = _content_scores(model, user_id)
    popularity_component = _normalize_scores(model.mean_item_ratings)

    weights = model.config["weights"]
    final_scores = np.zeros(num_items, dtype=np.float32)

    if collab_component is not None:
        final_scores += weights["collaborative"] * collab_component
    if content_component is not None:
        final_scores += weights["content"] * content_component
    if popularity_component is not None:
        final_scores += weights["popularity"] * popularity_component

    already_rated: Sequence[str] = ()
    if not include_rated:
        already_rated = model.feedback.loc[model.feedback["userId"] == user_id, "destinationId"].tolist()

    lex_keys = (model.rating_counts, model.mean_item_ratings, final_scores)
    ranked_indices = np.lexsort(lex_keys)[::-1]
    recommendations: List[Dict[str, object]] = []

    for idx in ranked_indices:
        dest_id = model.destinations.iloc[idx]["destinationId"]
        if not include_rated and dest_id in already_rated:
            continue
        recommendation = {
            "destinationId": dest_id,
            "score": float(final_scores[idx]),
            "category": model.destinations.iloc[idx].get("category"),
            "province": model.destinations.iloc[idx].get("province"),
            "averageRating": float(model.destinations.iloc[idx].get("averageRating", np.nan)),
            "description": model.destinations.iloc[idx].get("description", ""),
            "components": {
                "collaborative": float(collab_component[idx]) if collab_component is not None else None,
                "content": float(content_component[idx]) if content_component is not None else None,
                "popularity": float(popularity_component[idx]) if popularity_component is not None else None,
            },
        }
        recommendations.append(recommendation)
        if len(recommendations) >= top_n:
            break

    return recommendations


def save_model(model: HybridRecommenderModel, artifacts_dir: str | Path = Path("artifacts/recommender")) -> Path:
    """Persist the recommender artifacts to disk."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifacts_dir / "hybrid_recommender.joblib"
    joblib.dump(model.to_payload(), target_path)
    return target_path


def load_model(artifacts_dir: str | Path = Path("artifacts/recommender")) -> HybridRecommenderModel:
    """Load a previously trained recommender from disk."""
    artifacts_dir = Path(artifacts_dir)
    target_path = artifacts_dir / "hybrid_recommender.joblib"
    payload = joblib.load(target_path)
    return HybridRecommenderModel(**payload)


def _load_dataframe(path: str | Path, *, required_columns: Iterable[str]) -> pd.DataFrame:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find data file at '{resolved_path}'.")
    df = pd.read_csv(resolved_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"File '{resolved_path}' is missing required columns: {missing}.")
    return df


def _resolve_weights(user_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    default = {"collaborative": 0.5, "content": 0.4, "popularity": 0.1}
    if not user_weights:
        return default
    combined = default.copy()
    for key, value in user_weights.items():
        if key in combined and value >= 0:
            combined[key] = float(value)
    total = sum(combined.values())
    if total == 0:
        return default
    return {key: value / total for key, value in combined.items()}


def _normalize_scores(scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if scores is None or len(scores) == 0:
        return None
    array = np.asarray(scores, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array)
    filtered = array[finite_mask]
    min_val = filtered.min()
    max_val = filtered.max()
    if np.isclose(max_val, min_val):
        normalized = np.zeros_like(array)
        normalized[finite_mask] = 0.5
        normalized[~finite_mask] = 0.0
        return normalized
    normalized = np.zeros_like(array)
    normalized[finite_mask] = (filtered - min_val) / (max_val - min_val)
    return normalized


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2 uses 'sparse'
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _collaborative_scores(model: HybridRecommenderModel, user_id: str) -> Optional[np.ndarray]:
    if model.collaborative_model is None or model.user_factors is None:
        return None
    user_idx = model.user_index.get(user_id)
    if user_idx is None:
        return None
    raw_scores = model.user_factors[user_idx] @ model.collaborative_model.components_
    return _normalize_scores(raw_scores)


def _content_scores(model: HybridRecommenderModel, user_id: str) -> Optional[np.ndarray]:
    profile = _build_user_profile(model, user_id)
    if profile is None:
        return None
    profile_norm = np.linalg.norm(profile)
    if profile_norm == 0.0:
        return None
    similarity = model.content_matrix @ profile
    similarity /= model.content_norms * profile_norm
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity = (similarity + 1.0) / 2.0
    return similarity.astype(np.float32)


def _build_user_profile(model: HybridRecommenderModel, user_id: str) -> Optional[np.ndarray]:
    positive_threshold = model.config["positive_feedback_threshold"]
    min_items = model.config["min_profile_items"]

    user_feedback = model.feedback.loc[model.feedback["userId"] == user_id]
    if not user_feedback.empty:
        profile = _profile_from_feedback(model, user_feedback, positive_threshold, min_items)
        if profile is not None:
            return profile

    user_row = model.users.loc[model.users["userId"] == user_id]
    if user_row.empty:
        return None

    favorite_category = user_row.iloc[0].get("favoriteCategory", "unknown") or "unknown"
    favorite_province = user_row.iloc[0].get("favoriteProvince", "unknown") or "unknown"
    synthetic_text = f"{favorite_category} {favorite_province}"
    desc_vector = model.description_vectorizer.transform([synthetic_text]).toarray()
    category_vector = model.category_encoder.transform([[favorite_category]])
    province_vector = model.province_encoder.transform([[favorite_province]])
    average_rating = float(model.destinations["averageRating"].mean())
    rating_vector = model.rating_scaler.transform([[average_rating]])

    profile = np.hstack([desc_vector, category_vector, province_vector, rating_vector]).astype(np.float32)
    return profile.flatten()


def _profile_from_feedback(
    model: HybridRecommenderModel,
    user_feedback: pd.DataFrame,
    positive_threshold: float,
    min_items: int,
) -> Optional[np.ndarray]:
    sorted_feedback = user_feedback.sort_values("rating", ascending=False)
    positive_feedback = sorted_feedback[sorted_feedback["rating"] >= positive_threshold]
    if positive_feedback.empty:
        positive_feedback = sorted_feedback.head(min_items)
    if positive_feedback.empty:
        return None

    vectors: List[np.ndarray] = []
    weights: List[float] = []
    for _, row in positive_feedback.iterrows():
        dest_id = row["destinationId"]
        dest_idx = model.destination_index.get(dest_id)
        if dest_idx is None:
            continue
        vectors.append(model.content_matrix[dest_idx])
        weights.append(float(row["rating"]))

    if not vectors:
        return None

    vectors_array = np.vstack(vectors)
    weights_array = np.asarray(weights, dtype=np.float32)
    if weights_array.sum() == 0:
        weights_array = np.ones_like(weights_array) / len(weights_array)
    else:
        weights_array = weights_array / weights_array.sum()
    profile = np.average(vectors_array, axis=0, weights=weights_array)
    return profile.astype(np.float32)