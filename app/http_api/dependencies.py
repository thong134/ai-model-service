from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..itinerary.travel_service import TravelService


@lru_cache(maxsize=1)
def get_travel_service() -> TravelService:
    data_dir = Path(os.getenv("TRAVEL_SERVICE_DATA_DIR", "data"))
    artifacts_dir = Path(os.getenv("TRAVEL_SERVICE_ARTIFACTS_DIR", "artifacts/recommender"))
    speed_value = os.getenv("TRAVEL_SERVICE_AVG_SPEED_KMH", "40.0")
    try:
        average_speed = float(speed_value)
    except ValueError:
        average_speed = 40.0

    return TravelService(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        average_speed_kmh=average_speed,
    )
