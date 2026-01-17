from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

@dataclass
class DestinationRecommender:
    destinations: pd.DataFrame
    config: Dict[str, float]

    def __post_init__(self):
        # Normalize favorite times and ratings for scoring
        self.scaler = MinMaxScaler()
        if not self.destinations.empty:
            self.destinations['norm_favorites'] = self.scaler.fit_transform(
                self.destinations[['favouriteTimes']].fillna(0)
            )
            # Normalize rating (0-5) to 0-1
            self.destinations['norm_rating'] = self.destinations['averageRating'].fillna(0) / 5.0
        else:
            self.destinations['norm_favorites'] = 0.0
            self.destinations['norm_rating'] = 0.0
            
        # Load TF-IDF artifacts
        import joblib
        try:
            self.vectorizer = joblib.load('artifacts/dest_tfidf_vectorizer.joblib')
            self.tfidf_matrix = joblib.load('artifacts/dest_tfidf_matrix.joblib')
            self.dest_ids = pd.read_csv('artifacts/dest_ids.csv')['destinationId'].astype(str).tolist()
            print("  ✓ Hybrid Recommendation Artifacts loaded")
        except Exception as e:
            print(f"  ⚠ TF-IDF artifacts not loaded: {e}")
            self.vectorizer = None
            self.tfidf_matrix = None

    def recommend(
        self,
        user_hobbies: List[str],
        user_favorites: List[str], # List of favorite destination IDs
        history_profile: Dict[str, int] = None, # Category frequencies from TravelRoute
        engagement_profile: Dict[str, int] = None, # Category frequencies from Feedbacks
        province: Optional[str] = None,
        top_n: int = 50,
        offset: int = 0
    ) -> List[Dict[str, object]]:
        
        # Mapping Hobby/Internal terms to Vietnamese CSV Categories
        # Consistent with DB categories: Thiên nhiên, Lịch sử, Giải trí, Công trình, Văn hóa, Phiêu lưu, Biển, Núi
        category_map = {
            'Thiên nhiên': ['Thiên nhiên'],
            'Văn hóa': ['Văn hóa'],
            'Lịch sử': ['Lịch sử'],
            'Giải trí': ['Giải trí'],
            'Công trình': ['Công trình'],
            'Biển': ['Biển'],
            'Núi': ['Núi'],
            'Phiêu lưu': ['Phiêu lưu'],
            # English Hobbies (from User Profile)
            'Adventure': ['Núi', 'Phiêu lưu'],
            'Relaxation': ['Biển', 'Thiên nhiên'],
            'Culture&History': ['Lịch sử', 'Văn hóa', 'Công trình'],
            'Entertainment': ['Giải trí'],
            'Nature': ['Thiên nhiên', 'Núi', 'Biển'],
            'Beach&Islands': ['Biển'],
            'Mountain&Forest': ['Núi', 'Thiên nhiên'],
            'Photography': ['Biển', 'Núi', 'Lịch sử'],
            'Foods&Drinks': ['Giải trí']
        }

        # Normalize profiles using the map
        def normalize_profile(profile):
            if not profile: return {}
            normalized = {}
            for cat_name, count in profile.items():
                mapped_cats = category_map.get(cat_name, [cat_name])
                for mc in mapped_cats:
                    normalized[mc] = normalized.get(mc, 0) + (count / len(mapped_cats))
            return normalized

        norm_history = normalize_profile(history_profile)
        norm_engagement = normalize_profile(engagement_profile)
        
        # Map user hobbies to target categories
        target_categories = set()
        for h in user_hobbies:
            for mc in category_map.get(h, [h]):
                target_categories.add(mc)

        candidates = self.destinations.copy()
        
        if province:
            candidates = candidates[candidates['province'].str.lower() == province.lower()]
            
        # FILTER: Exclude already favorited items to prioritize discovery
        if user_favorites:
            candidates = candidates[~candidates['destinationId'].astype(str).isin(user_favorites)]
            
        if candidates.empty:
            return []

        # 1. Hobby Match Score
        candidates['score_hobby'] = candidates['category'].apply(
            lambda x: 1.0 if x in target_categories else 0.0
        )

        # 2. History & Engagement Profile Scoring
        def get_profile_score(cat, profile):
            if not profile: return 0.0
            total = sum(profile.values())
            if total == 0: return 0.0
            return profile.get(cat, 0) / total

        candidates['score_history'] = candidates['category'].apply(
            lambda x: get_profile_score(x, norm_history)
        )
        candidates['score_engagement'] = candidates['category'].apply(
            lambda x: get_profile_score(x, norm_engagement)
        )

        # 3. Quality & Social Proof (NEW)
        # Using pre-calculated normalized columns
        candidates['score_rating'] = candidates['norm_rating']
        candidates['score_popularity'] = candidates['norm_favorites']

        # 4. Location Correlation (Calculated but weight is low/zero if not used)
        fav_provinces = self.destinations[
            self.destinations['destinationId'].astype(str).isin(user_favorites)
        ]['province'].unique()
        candidates['score_location'] = candidates['province'].apply(
            lambda x: 1.0 if x in fav_provinces else 0.0
        )

        # 5. TF-IDF Similarity
        candidates['score_tfidf'] = 0.0
        if self.vectorizer and self.tfidf_matrix is not None:
            # Better Interest Text: Use target categories + hobbies
            interest_text = " ".join(list(target_categories) + user_hobbies)
            
            # Boost text from Behavior Profiles
            if history_profile:
                # Add categories from history 
                interest_text += " " + " ".join([cat for cat, count in history_profile.items() for _ in range(min(count, 3))])
            
            if engagement_profile:
                interest_text += " " + " ".join([cat for cat, count in engagement_profile.items() for _ in range(min(count, 3))])

            # CRITICAL: Add Favorites info to find SIMILAR items
            # Since we filtered out the favorites from candidates, we use them here 
            # effectively as the query to find similar items.
            if user_favorites:
                fav_info = self.destinations[self.destinations['destinationId'].astype(str).isin(user_favorites)]
                if not fav_info.empty:
                    fav_names = " ".join(fav_info['name'].fillna(''))
                    fav_cats = " ".join(fav_info['category'].fillna(''))
                    interest_text += f" {fav_names} {fav_cats}"
            
            if interest_text.strip():
                user_vec = self.vectorizer.transform([interest_text])
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
                sim_map = dict(zip(self.dest_ids, similarities))
                candidates['score_tfidf'] = candidates['destinationId'].astype(str).map(sim_map).fillna(0.0)

        # Weights (V5.1 - Unified Discovery)
        w_hobby = 0.15
        w_history = 0.35 # (30% History + 5% Engagement merged)
        w_rating = 0.15
        w_popularity = 0.10
        w_tfidf = 0.25
        w_location = 0.05

        # Calculate final score with fillna to be safe
        candidates['score'] = (
            w_hobby * candidates['score_hobby'].fillna(0.0) +
            w_history * candidates['score_history'].fillna(0.0) +
            w_rating * candidates['score_rating'].fillna(0.0) +
            w_popularity * candidates['score_popularity'].fillna(0.0) +
            w_tfidf * candidates['score_tfidf'].fillna(0.0) +
            w_location * candidates['score_location'].fillna(0.0)
        )

        # Explicit Sort: Total Score Descending (Tie-break by ID Ascending)
        candidates = candidates.sort_values(by=['score', 'destinationId'], ascending=[False, True])
        
        # Apply pagination
        candidates = candidates.iloc[offset : offset + top_n]
        
        recommendations = []
        
        recommendations = []
        for _, row in candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "category": row['category'],
                "score": float(row['score']),
                "reason": {
                    "hobby_match": float(w_hobby * row['score_hobby']),
                    "history_match": float(w_history * row['score_history']),
                    "quality_match": float(w_rating * row['score_rating']),
                    "popularity_match": float(w_popularity * row['score_popularity']),
                    "semantic_match": float(w_tfidf * row['score_tfidf']),
                    "location_match": float(w_location * row['score_location'])
                }
            })
            
        return recommendations

    def inspect(
        self,
        user_hobbies: List[str],
        user_favorites: List[str],
        history_profile: Dict[str, int] = None,
        engagement_profile: Dict[str, int] = None,
        province: Optional[str] = None,
        top_n: int = 50,
        offset: int = 0
    ) -> Dict[str, object]:
        
        # Consistent Mapping with recommend()
        category_map = {
            'Thiên nhiên': ['Thiên nhiên'],
            'Văn hóa': ['Văn hóa'],
            'Lịch sử': ['Lịch sử'],
            'Giải trí': ['Giải trí'],
            'Công trình': ['Công trình'],
            'Biển': ['Biển'],
            'Núi': ['Núi'],
            'Phiêu lưu': ['Phiêu lưu'],
            'Adventure': ['Núi', 'Phiêu lưu'],
            'Relaxation': ['Biển', 'Thiên nhiên'],
            'Culture&History': ['Lịch sử', 'Văn hóa', 'Công trình'],
            'Entertainment': ['Giải trí'],
            'Nature': ['Thiên nhiên', 'Núi', 'Biển'],
            'Beach&Islands': ['Biển'],
            'Mountain&Forest': ['Núi', 'Thiên nhiên'],
            'Photography': ['Biển', 'Núi', 'Lịch sử'],
            'Foods&Drinks': ['Giải trí']
        }

        def normalize_profile(profile):
            if not profile: return {}
            normalized = {}
            for cat_name, count in profile.items():
                mapped_cats = category_map.get(cat_name, [cat_name])
                for mc in mapped_cats:
                    normalized[mc] = normalized.get(mc, 0) + (count / len(mapped_cats))
            return normalized

        norm_history = normalize_profile(history_profile)
        norm_engagement = normalize_profile(engagement_profile)
        
        target_categories = set()
        for h in user_hobbies:
            for mc in category_map.get(h, [h]):
                target_categories.add(mc)

        candidates = self.destinations.copy()
        
        if province:
            candidates = candidates[candidates['province'].str.lower() == province.lower()]
            
        # FILTER: Exclude favorites
        if user_favorites:
            candidates = candidates[~candidates['destinationId'].astype(str).isin(user_favorites)]

        if candidates.empty:
            return {"error": "No candidates found for criteria", "debug_info": { "province": province }}

        # Recalculate component scores
        candidates['score_hobby'] = candidates['category'].apply(
            lambda x: 1.0 if x in target_categories else 0.0
        )
        
        def get_profile_score(cat, profile):
            if not profile: return 0.0
            total = sum(profile.values())
            if total == 0: return 0.0
            return profile.get(cat, 0) / total

        candidates['score_history'] = candidates['category'].apply(
            lambda x: get_profile_score(x, norm_history)
        )
        candidates['score_engagement'] = candidates['category'].apply(
            lambda x: get_profile_score(x, norm_engagement)
        )
        
        # New Components
        candidates['score_rating'] = candidates['norm_rating']
        candidates['score_popularity'] = candidates['norm_favorites']
        
        # Location (For debug info)
        fav_provinces = self.destinations[
            self.destinations['destinationId'].astype(str).isin(user_favorites)
        ]['province'].unique()
        candidates['score_location'] = candidates['province'].apply(
            lambda x: 1.0 if x in fav_provinces else 0.0
        )

        candidates['score_tfidf'] = 0.0
        if self.vectorizer and self.tfidf_matrix is not None:
             interest_text = " ".join(list(target_categories) + user_hobbies)
             if history_profile:
                # Add categories from history 
                interest_text += " " + " ".join([cat for cat, count in history_profile.items() for _ in range(min(count, 3))])
             
             # Favorites Similarity
             if user_favorites:
                fav_info = self.destinations[self.destinations['destinationId'].astype(str).isin(user_favorites)]
                if not fav_info.empty:
                    fav_names = " ".join(fav_info['name'].fillna(''))
                    fav_cats = " ".join(fav_info['category'].fillna(''))
                    interest_text += f" {fav_names} {fav_cats}"

             if interest_text.strip():
                user_vec = self.vectorizer.transform([interest_text])
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
                sim_map = dict(zip(self.dest_ids, similarities))
                candidates['score_tfidf'] = candidates['destinationId'].astype(str).map(sim_map).fillna(0.0)

        # Weights (V5.1 - Unified Discovery)
        w_hobby = 0.15
        w_history = 0.35
        w_rating = 0.15
        w_popularity = 0.10
        w_tfidf = 0.25
        w_location = 0.05

        candidates['score_hobby_weighted'] = w_hobby * candidates['score_hobby'].fillna(0.0)
        candidates['score_history_weighted'] = w_history * candidates['score_history'].fillna(0.0)
        candidates['score_rating_weighted'] = w_rating * candidates['score_rating'].fillna(0.0)
        candidates['score_popularity_weighted'] = w_popularity * candidates['score_popularity'].fillna(0.0)
        candidates['score_tfidf_weighted'] = w_tfidf * candidates['score_tfidf'].fillna(0.0)
        candidates['score_location_weighted'] = w_location * candidates['score_location'].fillna(0.0)

        candidates['score'] = (
            w_hobby * candidates['score_hobby'].fillna(0.0) +
            w_history * candidates['score_history'].fillna(0.0) +
            w_rating * candidates['score_rating'].fillna(0.0) +
            w_popularity * candidates['score_popularity'].fillna(0.0) +
            w_tfidf * candidates['score_tfidf'].fillna(0.0) +
            w_location * candidates['score_location'].fillna(0.0)
        )

        # Explicit Sort: Total Score Descending
        candidates = candidates.sort_values(by=['score', 'destinationId'], ascending=[False, True])
        top_candidates = candidates.iloc[offset : offset + top_n]
        
        recommendations = []
        for _, row in top_candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "category": row['category'],
                "score": float(row['score']),
                "reason": {
                    "hobby_match": float(w_hobby * row['score_hobby']),
                    "history_match": float(w_history * row['score_history']),
                    "quality_match": float(w_rating * row['score_rating']),
                    "popularity_match": float(w_popularity * row['score_popularity']),
                    "semantic_match": float(w_tfidf * row['score_tfidf']),
                    "location_match": float(w_location * row['score_location'])
                }
            })
            
        return {
            "debug_info": {
                "inputs": {
                    "hobbies": user_hobbies,
                    "favorites": user_favorites,
                    "history_profile": history_profile,
                    "engagement_profile": engagement_profile,
                    "province": province
                },
                "weights_config": {
                    "w_hobby": w_hobby,
                    "w_history": w_history,
                    "w_rating": w_rating,
                    "w_popularity": w_popularity,
                    "w_tfidf": w_tfidf,
                    "w_location": w_location
                }
            },
            "recommendations": recommendations,
            "candidates_count": len(candidates)
        }

def train_model(
    destinations_path: str | Path = Path("data/destinations.csv"),
    weights: Optional[Dict[str, float]] = None
) -> DestinationRecommender:
    
    if not Path(destinations_path).exists():
        # Fallback for empty init
        df = pd.DataFrame(columns=[
            'destinationId', 'name', 'province', 'category', 
            'averageRating', 'favouriteTimes', 'latitude', 'longitude'
        ])
    else:
        df = pd.read_csv(destinations_path)
        
    # Ensure columns exist
    required = ['averageRating', 'favouriteTimes']
    for col in required:
        if col not in df.columns:
            df[col] = 0

    default_weights = {
        'w_hobby': 0.5,
        'w_social': 0.3,
        'w_rating': 0.2,
        'w_personal': 0.1 # Boost known favorites slightly
    }
    
    if weights:
        default_weights.update(weights)

    return DestinationRecommender(destinations=df, config=default_weights)
