from __future__ import annotations

from flask import Flask
from flasgger import Swagger

from .api import SWAGGER_TEMPLATE, register_routes
from .http_api import create_fastapi_app
from .moderation.config import AppConfig
from .moderation.service import ReviewService
from .itinerary.travel_service import RoutePlan, TravelService


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(__name__)
    Swagger(app, template=SWAGGER_TEMPLATE)
    service = ReviewService(config)
    register_routes(app, service)
    return app


__all__ = [
    "create_app",
    "create_fastapi_app",
    "AppConfig",
    "ReviewService",
    "TravelService",
    "RoutePlan",
]
