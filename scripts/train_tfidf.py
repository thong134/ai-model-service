import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def train_tfidf(data_path="data/destinations.csv", output_dir="artifacts"):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Combine textual fields available in the CSV
    text_cols = ['name', 'province', 'category', 'description']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''
            
    df['text_content'] = df['name'].astype(str) + " " + \
                        df['province'].astype(str) + " " + \
                        df['category'].astype(str) + " " + \
                        df['description'].astype(str)
    
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words='english', # Basic for Eng, for Viet we rely on keywords
        ngram_range=(1, 2),
        max_features=5000
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['text_content'])
    
    # Create artifacts directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the artifacts
    joblib.dump(vectorizer, os.path.join(output_dir, "dest_tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(output_dir, "dest_tfidf_matrix.joblib"))
    
    # Also save the destination IDs to ensure alignment during inference
    df[['destinationId']].to_csv(os.path.join(output_dir, "dest_ids.csv"), index=False)
    
    print(f"Done! Artifacts saved to {output_dir}")
    print(f"Matrix shape: {tfidf_matrix.shape}")

if __name__ == "__main__":
    train_tfidf()
