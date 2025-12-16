from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_travel_service
from ..schemas import RouteRequest, RouteResponse
from ...itinerary.travel_service import TravelService

router = APIRouter(tags=["Routes"])


def _build_start_payload(request: RouteRequest) -> object:
    if request.start_destination_id:
        return request.start_destination_id
    if request.start_coordinates:
        latitude = request.start_coordinates.latitude
        longitude = request.start_coordinates.longitude
        label = request.start_label or "Start"
        return {
            "id": label or "__start__",
            "label": label,
            "coords": (latitude, longitude),
        }
    raise HTTPException(status_code=400, detail="Provide startDestinationId or startCoordinates.")


@router.post("/route", response_model=RouteResponse)
def generate_route(
    payload: RouteRequest,
    service: TravelService = Depends(get_travel_service),
) -> RouteResponse:
    start_payload = _build_start_payload(payload)

    try:
        plan = service.plan_route(
            user_id=payload.user_id,
            province=payload.province,
            start=start_payload,
            destination_ids=payload.destination_ids,
            max_time_hours=payload.max_time,
            top_n=payload.top_n,
            include_rated=payload.include_rated,
            random_seed=payload.random_seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RouteResponse(
        summary=plan.summary,
        itinerary=plan.itinerary,
        visualization=plan.visualization,
    )
