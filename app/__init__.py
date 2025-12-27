from __future__ import annotations

# Clean exports for the new AI service structure
from .moderation.service import ReviewService
from .recommendation.destination import DestinationRecommender, train_model as load_dest_model
from .recommendation.route import RouteRecommender
from .vision.classifier import PlaceClassifier

__all__ = [
    "ReviewService",
    "DestinationRecommender",
    "load_dest_model",
    "RouteRecommender",
    "PlaceClassifier",
]
