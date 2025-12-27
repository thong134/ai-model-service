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

    def recommend(
        self,
        user_hobbies: List[str],
        user_favorites: List[str], # List of favorite destination IDs
        province: Optional[str] = None,
        top_n: int = 10
    ) -> List[Dict[str, object]]:
        
        candidates = self.destinations.copy()
        
        if province:
             # Case-insensitive filtering
            candidates = candidates[candidates['province'].str.lower() == province.lower()]
            
        if candidates.empty:
            return []

        # 1. Hobby Match Score (0 or 1)
        # Check if destination category is in user hobbies
        # Assuming 'category' column exists and user_hobbies are categories
        candidates['hobby_match'] = candidates['category'].apply(
            lambda x: 1.0 if x in user_hobbies else 0.0
        )

        # 2. Favorite Score (Social Proof) -> Already normalized as 'norm_favorites'

        # 3. Rating Score (Quality) -> Already normalized as 'norm_rating'
        
        # 4. User Personal Favorite Boost
        # If user already favorited this place, maybe boost it? 
        # OR usually we recommend new things. 
        # Let's say we recommend "suitable" things. If they favorited it, it's suitable.
        candidates['is_user_fav'] = candidates['destinationId'].astype(str).isin(user_favorites).astype(float)

        # Calculate Final Weighted Score
        w_hobby = self.config.get('w_hobby', 0.4)
        w_social = self.config.get('w_social', 0.2) # General popularity
        w_rating = self.config.get('w_rating', 0.2) # Quality
        w_personal = self.config.get('w_personal', 0.2) # Personal preference

        candidates['score'] = (
            w_hobby * candidates['hobby_match'] +
            w_social * candidates['norm_favorites'] +
            w_rating * candidates['norm_rating'] + 
            w_personal * candidates['is_user_fav']
        )

        # Sort and return
        candidates = candidates.sort_values('score', ascending=False).head(top_n)
        
        recommendations = []
        for _, row in candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "score": float(row['score']),
                "reason": {
                    "hobby_match": bool(row['hobby_match']),
                    "popularity": float(row['norm_favorites']),
                    "rating": float(row['averageRating'])
                }
            })
            
        return recommendations

    def inspect(
        self,
        user_hobbies: List[str],
        user_favorites: List[str],
        province: Optional[str] = None,
        top_n: int = 10
    ) -> Dict[str, object]:
        
        candidates = self.destinations.copy()
        
        if province:
            candidates = candidates[candidates['province'].str.lower() == province.lower()]
            
        if candidates.empty:
            return {"error": "No candidates found for criteria", "debug_info": { "province": province }}

        # Recalculate component scores (same as recommend)
        candidates['hobby_match'] = candidates['category'].apply(
            lambda x: 1.0 if x in user_hobbies else 0.0
        )
        candidates['is_user_fav'] = candidates['destinationId'].astype(str).isin(user_favorites).astype(float)

        w_hobby = self.config.get('w_hobby', 0.4)
        w_social = self.config.get('w_social', 0.2)
        w_rating = self.config.get('w_rating', 0.2)
        w_personal = self.config.get('w_personal', 0.2)

        candidates['score_hobby'] = w_hobby * candidates['hobby_match']
        candidates['score_social'] = w_social * candidates['norm_favorites'] 
        candidates['score_rating'] = w_rating * candidates['norm_rating']
        candidates['score_personal'] = w_personal * candidates['is_user_fav']

        candidates['score'] = (
            candidates['score_hobby'] +
            candidates['score_social'] +
            candidates['score_rating'] + 
            candidates['score_personal']
        )

        top_candidates = candidates.sort_values('score', ascending=False).head(top_n)
        
        recommendations = []
        for _, row in top_candidates.iterrows():
            recommendations.append({
                "destinationId": str(row['destinationId']),
                "name": row['name'],
                "category": row['category'],
                "total_score": float(row['score']),
                "components": {
                    "hobby": {
                        "raw_match": bool(row['hobby_match']),
                        "weight": w_hobby,
                        "weighted_score": float(row['score_hobby'])
                    },
                    "social": {
                        "norm_value": float(row['norm_favorites']),
                        "weight": w_social,
                        "weighted_score": float(row['score_social'])
                    },
                    "rating": {
                        "norm_value": float(row['norm_rating']),
                        "weight": w_rating,
                        "weighted_score": float(row['score_rating'])
                    },
                    "personal": {
                        "is_fav": bool(row['is_user_fav']),
                        "weight": w_personal,
                        "weighted_score": float(row['score_personal'])
                    }
                }
            })
            
        return {
            "debug_info": {
                "inputs": {
                    "hobbies": user_hobbies,
                    "favorites": user_favorites,
                    "province": province
                },
                "weights_config": self.config
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
