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
        
        # Mapping Vietnamese/Hobby terms to CSV Categories
        # CSV Categories are: Beach, Historical, Mountain, Forest, Temple, Urban
        category_map = {
            'Thiên nhiên': ['Forest', 'Mountain', 'Beach'],
            'Văn hóa': ['Historical', 'Temple'],
            'Lịch sử': ['Historical'],
            'Giải trí': ['Urban'],
            'Công trình': ['Temple', 'Urban'],
            'Biển': ['Beach'],
            'Núi': ['Mountain'],
            # Hobbies
            'Adventure': ['Mountain', 'Forest'],
            'Relaxation': ['Beach', 'Forest'],
            'Culture&History': ['Historical', 'Temple'],
            'Entertainment': ['Urban'],
            'Nature': ['Forest', 'Mountain', 'Beach'],
            'Beach&Islands': ['Beach'],
            'Mountain&Forest': ['Mountain', 'Forest'],
            'Photography': ['Beach', 'Mountain', 'Historical'],
            'Foods&Drinks': ['Urban']
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

        # 3. Personal Favorite Boost
        candidates['is_user_fav'] = candidates['destinationId'].astype(str).isin(user_favorites).astype(float)

        # 4. Location Correlation
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
                # Add categories from history (weighted by count?)
                interest_text += " " + " ".join([cat for cat, count in history_profile.items() for _ in range(min(count, 3))])
            
            if engagement_profile:
                interest_text += " " + " ".join([cat for cat, count in engagement_profile.items() for _ in range(min(count, 3))])

            # CRITICAL: Add Favorites info to find SIMILAR items
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

        # Weights (V4.1 - Further boost Behavioral & Intent)
        w_hobby = 0.1
        w_history = 0.35 # +0.05
        w_engagement = 0.25 # +0.05
        w_personal = 0.4 # +0.15 (Makes favorite jump to top instantly)
        w_tfidf = 0.1
        w_location = 0.1 # +0.05

        candidates['score'] = (
            w_hobby * candidates['score_hobby'] +
            w_history * candidates['score_history'] +
            w_engagement * candidates['score_engagement'] +
            w_personal * candidates['is_user_fav'] +
            w_tfidf * candidates['score_tfidf'] +
            w_location * candidates['score_location']
        )

        # Sort and return with stable tie-breaker
        candidates = candidates.sort_values(['score', 'destinationId'], ascending=[False, True])
        
        # Apply pagination
        candidates = candidates.iloc[offset : offset + top_n]
        
        recommendations = []
        
        recommendations = []
        for _, row in candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "score": float(row['score']),
                "reason": {
                    "hobby_match": bool(row['score_hobby'] > 0),
                    "history_match": float(row['score_history']),
                    "engagement_match": float(row['score_engagement']),
                    "is_favorite": bool(row['is_user_fav']),
                    "semantic_score": float(row['score_tfidf'])
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
        
        # Consistent mapping with recommend()
        category_map = {
            'Thiên nhiên': ['Forest', 'Mountain', 'Beach'],
            'Văn hóa': ['Historical', 'Temple'],
            'Lịch sử': ['Historical'],
            'Giải trí': ['Urban'],
            'Công trình': ['Temple', 'Urban'],
            'Biển': ['Beach'],
            'Núi': ['Mountain'],
            'Adventure': ['Mountain', 'Forest'],
            'Relaxation': ['Beach', 'Forest'],
            'Culture&History': ['Historical', 'Temple'],
            'Entertainment': ['Urban'],
            'Nature': ['Forest', 'Mountain', 'Beach'],
            'Beach&Islands': ['Beach'],
            'Mountain&Forest': ['Mountain', 'Forest'],
            'Photography': ['Beach', 'Mountain', 'Historical'],
            'Foods&Drinks': ['Urban']
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
        candidates['is_user_fav'] = candidates['destinationId'].astype(str).isin(user_favorites).astype(float)
        
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
                interest_text += " " + " ".join(history_profile.keys())
            
             if interest_text.strip():
                user_vec = self.vectorizer.transform([interest_text])
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
                sim_map = dict(zip(self.dest_ids, similarities))
                candidates['score_tfidf'] = candidates['destinationId'].astype(str).map(sim_map).fillna(0.0)

        # Weights (V4.1)
        w_hobby = 0.1
        w_history = 0.35
        w_engagement = 0.25
        w_personal = 0.4
        w_tfidf = 0.1
        w_location = 0.1

        candidates['score_hobby_weighted'] = w_hobby * candidates['score_hobby']
        candidates['score_history_weighted'] = w_history * candidates['score_history']
        candidates['score_engagement_weighted'] = w_engagement * candidates['score_engagement']
        candidates['score_personal_weighted'] = w_personal * candidates['is_user_fav']
        candidates['score_tfidf_weighted'] = w_tfidf * candidates['score_tfidf']
        candidates['score_location_weighted'] = w_location * candidates['score_location']

        candidates['score'] = (
            candidates['score_hobby_weighted'] +
            candidates['score_history_weighted'] +
            candidates['score_engagement_weighted'] +
            candidates['score_personal_weighted'] +
            candidates['score_tfidf_weighted'] +
            candidates['score_location_weighted']
        )

        # Sort and paginate
        candidates = candidates.sort_values(['score', 'destinationId'], ascending=[False, True])
        top_candidates = candidates.iloc[offset : offset + top_n]
        
        recommendations = []
        for _, row in top_candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "category": row['category'],
                "total_score": float(row['score']),
                "components": {
                    "hobby": {
                        "raw_match": bool(row['score_hobby'] > 0),
                        "weight": w_hobby,
                        "weighted_score": float(row['score_hobby_weighted'])
                    },
                    "history": {
                        "freq": float(row['score_history']),
                        "weight": w_history,
                        "weighted_score": float(row['score_history_weighted'])
                    },
                    "engagement": {
                        "freq": float(row['score_engagement']),
                        "weight": w_engagement,
                        "weighted_score": float(row['score_engagement_weighted'])
                    },
                    "personal": {
                        "is_fav": bool(row['is_user_fav']),
                        "weight": w_personal,
                        "weighted_score": float(row['score_personal_weighted'])
                    },
                    "semantic": {
                        "tfidf_sim": float(row['score_tfidf']),
                        "weight": w_tfidf,
                        "weighted_score": float(row['score_tfidf_weighted'])
                    },
                    "location": {
                        "matched_fav_province": bool(row['score_location'] > 0),
                        "weight": w_location,
                        "weighted_score": float(row['score_location_weighted'])
                    }
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
                    "w_engagement": w_engagement,
                    "w_personal": w_personal,
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
