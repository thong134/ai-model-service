import pandas as pd
import random
import os

def generate_data():
    os.makedirs('data', exist_ok=True)
    
    # 1. Destinations
    provinces = ['Da Nang', 'Quang Nam', 'Hue', 'Ha Noi', 'Ho Chi Minh']
    categories = ['Beach', 'Mountain', 'Temple', 'Urban', 'Forest', 'Historical']
    
    destinations = []
    for i in range(1, 101):
        prov = random.choice(provinces)
        # Generate varied coords around a base for each province
        base_coords = {
            'Da Nang': (16.0544, 108.2022),
            'Quang Nam': (15.8801, 108.3380),
            'Hue': (16.4674, 107.5905),
            'Ha Noi': (21.0285, 105.8542),
            'Ho Chi Minh': (10.8231, 106.6297)
        }
        lat_base, lon_base = base_coords[prov]
        
        destinations.append({
            'destinationId': i,
            'name': f"Destination {i}",
            'province': prov,
            'category': random.choice(categories),
            'averageRating': round(random.uniform(3.0, 5.0), 1),
            'favouriteTimes': random.randint(0, 500),
            'latitude': lat_base + random.uniform(-0.05, 0.05),
            'longitude': lon_base + random.uniform(-0.05, 0.05),
            'description': f"This is a beautiful place in {prov}."
        })
        
    df_dest = pd.DataFrame(destinations)
    df_dest.to_csv('data/destinations.csv', index=False)
    print("Generated data/destinations.csv")

if __name__ == "__main__":
    generate_data()
