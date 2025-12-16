from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_travel_service
from ..schemas import RecommendationItem, RecommendationRequest, RecommendationResponse
from ...itinerary.travel_service import TravelService

router = APIRouter(tags=["Recommendations"])


@router.post("/recommendations", response_model=RecommendationResponse)
def generate_recommendations(
    payload: RecommendationRequest,
    service: TravelService = Depends(get_travel_service),
) -> RecommendationResponse:
    try:
        items = service.recommend_destinations(
            user_id=payload.user_id,
            top_n=payload.top_n,
            include_rated=payload.include_rated,
            province=payload.province,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    parsed: List[RecommendationItem] = [RecommendationItem.from_raw(item) for item in items]
    return RecommendationResponse(user_id=payload.user_id, total=len(parsed), destinations=parsed)
