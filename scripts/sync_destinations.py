#!/usr/bin/env python3
"""
Sync Destinations from Traveline Backend API to AI Model Service.
This script fetches real destination data and saves it locally for AI model consumption.

Usage:
    python scripts/sync_destinations.py [--api-url=http://localhost:3001]
"""

import argparse
import json
import os
import sys
import pandas as pd
import requests
from pathlib import Path


def sync_destinations(api_url: str, output_path: str = "data/destinations.csv"):
    """
    Fetch destinations from the backend API and save to CSV.
    """
    export_url = f"{api_url}/destinations/export"
    
    print(f"Fetching destinations from: {export_url}")
    
    try:
        response = requests.get(export_url, timeout=30)
        response.raise_for_status()
        
        destinations = response.json()
        
        if not destinations:
            print("Warning: No destinations returned from API")
            return False
        
        print(f"Received {len(destinations)} destinations")
        
        # Convert to DataFrame
        df = pd.DataFrame(destinations)
        
        # Ensure required columns exist
        required_columns = ['destinationId', 'name', 'province', 'category', 
                           'averageRating', 'favouriteTimes', 'latitude', 'longitude']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing column '{col}', creating empty")
                df[col] = '' if col in ['name', 'province', 'category'] else 0
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} destinations to {output_path}")
        
        # Print province distribution
        print("\nProvince distribution:")
        for prov, count in df['province'].value_counts().items():
            print(f"  - {prov}: {count} destinations")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to fetch from API: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def rebuild_tfidf_vectors(data_path: str = "data/destinations.csv"):
    """
    Rebuild TF-IDF vectors after data sync.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib
        
        print("\nRebuilding TF-IDF vectors...")
        
        df = pd.read_csv(data_path)
        
        # Create text corpus from descriptions and other fields
        df['text_corpus'] = (
            df['name'].fillna('') + ' ' +
            df['category'].fillna('') + ' ' +
            df['province'].fillna('') + ' ' +
            df.get('description', '').fillna('')
        )
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df['text_corpus'])
        
        # Save artifacts
        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(vectorizer, 'artifacts/dest_tfidf_vectorizer.joblib')
        joblib.dump(tfidf_matrix, 'artifacts/dest_tfidf_matrix.joblib')
        df[['destinationId']].to_csv('artifacts/dest_ids.csv', index=False)
        
        print(f"✓ TF-IDF vectors rebuilt ({tfidf_matrix.shape[1]} features)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to rebuild TF-IDF: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Sync destinations for AI models')
    parser.add_argument('--api-url', default='http://localhost:3001',
                        help='Backend API URL (default: http://localhost:3001)')
    parser.add_argument('--output', default='data/destinations.csv',
                        help='Output CSV path (default: data/destinations.csv)')
    parser.add_argument('--rebuild-tfidf', action='store_true',
                        help='Rebuild TF-IDF vectors after sync')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(" AI Model Service - Destination Data Sync")
    print("=" * 50)
    
    # Change to script's parent directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Sync destinations
    if sync_destinations(args.api_url, args.output):
        # Optionally rebuild TF-IDF
        if args.rebuild_tfidf:
            rebuild_tfidf_vectors(args.output)
        
        print("\n✓ Sync completed successfully!")
        print("\nIMPORTANT: Restart the AI service to load the new data:")
        print("  python app/api.py")
    else:
        print("\n✗ Sync failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
