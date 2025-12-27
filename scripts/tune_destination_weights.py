import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.recommendation.destination import DestinationRecommender
import argparse
from pathlib import Path

def tune_weights(dest_file: str):
    print("For this Heuristic Model, 'training' means validating score distribution based on weights.")
    print("This script simulates recommendations for a dummy user profile to sanity check scores.")
    
    # 1. Load Data
    df = pd.read_csv(dest_file)
    print(f"Loaded {len(df)} destinations.")
    
    # 2. Define Weights to Test (Hyperparameters)
    weights = {
        'w_hobby': 0.5,
        'w_social': 0.3,
        'w_rating': 0.1,
        'w_personal': 0.1
    }
    print(f"Testing weights: {weights}")
    
    recommender = DestinationRecommender(df, weights)
    
    # 3. Simulate User
    user_hobbies = ['Beach', 'Urban']
    user_favorites = ['1', '5'] # IDs
    province = 'Da Nang'
    
    print(f"User Hobbies: {user_hobbies}, Favorites IDs: {user_favorites}, Province: {province}")
    
    results = recommender.recommend(user_hobbies, user_favorites, province, top_n=5)
    
    print("\nTop 5 Recommendations:")
    for res in results:
        print(f"- {res['name']} (Score: {res['score']:.2f})")
        print(f"  Reasons: {res['reason']}")
        
    print("\nIf scores look balanced, these weights are good. Modify 'app/recommendation/destination.py' to make them permanent.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/destinations.csv')
    args = parser.parse_args()
    
    tune_weights(args.data)
