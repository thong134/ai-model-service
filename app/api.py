from __future__ import annotations
# Trigger reload
from flask import Flask, request, jsonify
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Global model cache (Lazy Loading)
_models = {
    "dest": None,
    "route": None,
    "vision": None,
    "moderation": None
}

@app.route('/', methods=['GET'])
def index():
    """Friendly root endpoint to confirm service is alive."""
    return jsonify({
        "service": "Traveline AI Model Service",
        "status": "online",
        "version": "1.1.0",
        "documentation": "/health",
        "endpoints": {
            "health": "/health",
            "reload": "/reload (POST)",
            "destination_recommendations": "/recommend/destinations (POST)",
            "route_recommendations": "/recommend/route (POST)",
            "text_moderation": "/moderation/predict (POST)",
            "vision_classification": "/vision/classify (POST)"
        },
        "message": "Welcome to the Traveline AI brain. Please use the documented API endpoints for interaction."
    }), 200

def get_dest_model():
    if _models["dest"] is None:
        print("Loading Destination Model (Lazy)...")
        try:
            from app.recommendation.destination import train_model as load_dest_model
            _models["dest"] = load_dest_model()
            print("  ✓ Destination Model loaded")
        except Exception as e:
            print(f"  ✗ Destination Model failed: {e}")
            _init_errors["dest"] = str(e)
    return _models["dest"]

def get_route_model():
    if _models["route"] is None:
        dm = get_dest_model()
        if dm:
            print("Loading Route Model (Lazy)...")
            try:
                from app.recommendation.route import RouteRecommender
                _models["route"] = RouteRecommender(dm)
                print("  ✓ Route Model loaded")
            except Exception as e:
                print(f"  ✗ Route Model failed: {e}")
                _init_errors["route"] = str(e)
        else:
             _init_errors["route"] = "Dependency 'dest' model failed to load."
    return _models["route"]

def get_vision_model():
    if _models["vision"] is None:
        print("Loading Place Classifier (Lazy)...")
        try:
            from app.vision.classifier import PlaceClassifier
            _models["vision"] = PlaceClassifier()
            print("  ✓ Place Classifier loaded")
        except Exception as e:
            print(f"  ✗ Place Classifier failed: {e}")
            _init_errors["vision"] = str(e)
    return _models["vision"]

def get_moderation_service():
    if _models["moderation"] is None:
        print("Loading Moderation Service (Lazy)...")
        try:
            from app.moderation.service import ReviewService
            svc = ReviewService()
            svc.load()
            _models["moderation"] = svc
            print("  ✓ Moderation Service loaded")
        except Exception as e:
            print(f"  ✗ Moderation Service failed: {e}")
            _init_errors["moderation"] = str(e)
    return _models["moderation"]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "services": {
            "destination": _models["dest"] is not None,
            "route": _models["route"] is not None,
            "vision": _models["vision"] is not None,
            "moderation": _models["moderation"] is not None
        },
        "errors": _init_errors
    })

@app.route('/reload', methods=['POST'])
def reload_models():
    print("Reloading models...")
    global _init_errors
    _init_errors = {}
    try:
        from app.recommendation.destination import train_model as load_dest_model
        _models["dest"] = load_dest_model()
        
        from app.recommendation.route import RouteRecommender
        if _models["dest"]:
            _models["route"] = RouteRecommender(_models["dest"])
            
        return jsonify({"status": "success", "message": "Models reloaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/moderation/predict/vi', methods=['POST'])
@app.route('/moderation/predict/en', methods=['POST'])
def moderate_text():
    lang = request.path.split('/')[-1]
    svc = get_moderation_service()
    if not svc:
        error_msg = _init_errors.get("moderation", "Moderation service not available")
        return jsonify({"error": error_msg}), 503
        
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    result = svc.score(text, lang=lang)
    return jsonify(result)

@app.route('/moderation/predict', methods=['POST'])
def moderate_text_legacy():
    # Legacy endpoint defaults to Vietnamese
    return moderate_text()

@app.route('/recommend/destinations', methods=['POST'])
def recommend_destinations():
    model = get_dest_model()
    if not model:
        return jsonify({"error": "Destination model not available"}), 503
        
    data = request.json
    hobbies = data.get('hobbies', [])
    favorites = data.get('favorites', [])
    history_profile = data.get('history_profile', {})
    engagement_profile = data.get('engagement_profile', {})
    province = data.get('province')
    top_n = data.get('limit', 50)
    offset = data.get('offset', 0)
    
    results = model.recommend(
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
    model = get_dest_model()
    if not model:
        return jsonify({"error": "Destination model not available"}), 503
        
    data = request.json
    hobbies = data.get('hobbies', [])
    favorites = data.get('favorites', [])
    history_profile = data.get('history_profile', {})
    engagement_profile = data.get('engagement_profile', {})
    province = data.get('province')
    top_n = data.get('limit', 50)
    offset = data.get('offset', 0)
    
    results = model.inspect(
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
    model = get_route_model()
    if not model:
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

    result = model.recommend_route(
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
    classifier = get_vision_model()
    if not classifier:
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
        
        result = classifier.predict(temp_path)
        return jsonify(result)
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def create_app():
    """Factory function for the Flask app."""
    return app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
