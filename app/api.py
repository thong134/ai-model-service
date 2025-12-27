from __future__ import annotations
# Trigger reload
from flask import Flask, request, jsonify
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Initialize model variables
dest_model = None
route_model = None
place_classifier = None
moderation_service = None

# Load Models Separately
print("Loading Destination Model...")
try:
    from app.recommendation.destination import train_model as load_dest_model
    dest_model = load_dest_model()
    print("  ✓ Destination Model loaded")
except Exception as e:
    print(f"  ✗ Destination Model failed: {e}")

print("Loading Route Model...")
try:
    from app.recommendation.route import RouteRecommender
    if dest_model:
        route_model = RouteRecommender(dest_model)
        print("  ✓ Route Model loaded")
    else:
        print("  ✗ Route Model skipped (requires Destination Model)")
except Exception as e:
    print(f"  ✗ Route Model failed: {e}")

print("Loading Place Classifier...")
try:
    from app.vision.classifier import PlaceClassifier
    place_classifier = PlaceClassifier()
    print("  ✓ Place Classifier loaded")
except Exception as e:
    print(f"  ✗ Place Classifier failed: {e}")

print("Loading Moderation Service...")
try:
    from app.moderation.service import ReviewService
    moderation_service = ReviewService()
    moderation_service.load()
    print("  ✓ Moderation Service loaded")
except Exception as e:
    print(f"  ✗ Moderation Service failed: {e}")

print("Model loading complete.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "services": {
            "destination": dest_model is not None,
            "route": route_model is not None,
            "vision": place_classifier is not None,
            "moderation": moderation_service is not None
        }
    })

@app.route('/moderation/predict', methods=['POST'])
def moderate_text():
    if not moderation_service:
        return jsonify({"error": "Moderation service not available"}), 503
        
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    result = moderation_service.score(text)
    return jsonify(result)

@app.route('/recommend/destinations', methods=['POST'])
def recommend_destinations():
    if not dest_model:
        return jsonify({"error": "Destination model not available"}), 503
        
    data = request.json
    hobbies = data.get('hobbies', [])
    favorites = data.get('favorites', [])
    history_profile = data.get('history_profile', {})
    engagement_profile = data.get('engagement_profile', {})
    province = data.get('province')
    top_n = data.get('limit', 50)
    offset = data.get('offset', 0)
    
    results = dest_model.recommend(
        user_hobbies=hobbies, 
        user_favorites=favorites, 
        history_profile=history_profile,
        engagement_profile=engagement_profile,
        province=province,
        top_n=top_n,
        offset=offset
    )
    return jsonify(results)

@app.route('/recommend/destinations/inspect', methods=['POST'])
def inspect_destinations():
    if not dest_model:
        return jsonify({"error": "Destination model not available"}), 503
        
    data = request.json
    hobbies = data.get('hobbies', [])
    favorites = data.get('favorites', [])
    history_profile = data.get('history_profile', {})
    engagement_profile = data.get('engagement_profile', {})
    province = data.get('province')
    top_n = data.get('limit', 50)
    offset = data.get('offset', 0)
    
    results = dest_model.inspect(
        user_hobbies=hobbies, 
        user_favorites=favorites, 
        history_profile=history_profile,
        engagement_profile=engagement_profile,
        province=province,
        top_n=top_n,
        offset=offset
    )
    return jsonify(results)

@app.route('/recommend/route', methods=['POST'])
def recommend_route():
    if not route_model:
        return jsonify({"error": "Route model not available"}), 503
        
    data = request.json
    hobbies = data.get('hobbies', [])
    favorites = data.get('favorites', [])
    province = data.get('province')
    start_date = data.get('startDate')
    end_date = data.get('endDate')
    
    start_lat = data.get('start_lat')
    start_long = data.get('start_long')
    start_loc = None
    if start_lat and start_long:
        start_loc = {"latitude": start_lat, "longitude": start_long}

    if not province:
        return jsonify({"error": "Province is required"}), 400
    if not start_date or not end_date:
        return jsonify({"error": "startDate and endDate are required"}), 400

    result = route_model.recommend_route(
        user_hobbies=hobbies,
        user_favorites=favorites,
        province=province,
        start_date=start_date,
        end_date=end_date,
        start_location=start_loc
    )
    return jsonify(result)

@app.route('/vision/classify', methods=['POST'])
def classify_place():
    if not place_classifier:
        return jsonify({"error": "Vision classifier not available"}), 503
    
    temp_path = None
    
    try:
        # Handle file upload
        if 'image' in request.files:
            image = request.files['image']
            temp_path = f"temp_{image.filename}"
            image.save(temp_path)
        # Handle URL
        elif request.json and request.json.get('imageUrl'):
            image_url = request.json.get('imageUrl')
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            temp_path = "temp_url_image.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
        else:
            return jsonify({"error": "No image provided (use 'image' file or 'imageUrl' in JSON)"}), 400
        
        result = place_classifier.predict(temp_path)
        return jsonify(result)
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
