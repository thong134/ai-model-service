from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScoreComponents(BaseModel):
    collaborative: Optional[float] = None
    content: Optional[float] = None
    popularity: Optional[float] = None

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class RecommendationItem(BaseModel):
    destination_id: str = Field(..., alias="destinationId")
    score: float
    category: Optional[str] = None
    province: Optional[str] = None
    average_rating: Optional[float] = Field(None, alias="averageRating")
    description: Optional[str] = None
    components: Optional[ScoreComponents] = None

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "RecommendationItem":
        try:
            return cls.parse_obj(data)  # type: ignore[attr-defined]
        except AttributeError:
            return cls.model_validate(data)  # type: ignore[attr-defined]


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., alias="userId")
    top_n: int = Field(10, alias="topN", ge=1, le=50)
    include_rated: bool = Field(False, alias="includeRated")
    province: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class RecommendationResponse(BaseModel):
    user_id: str = Field(..., alias="userId")
    total: int
    destinations: List[RecommendationItem]

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True
        json_encoders = {ScoreComponents: lambda v: v.dict(by_alias=True)}


class CoordinateInput(BaseModel):
    latitude: float
    longitude: float

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class RouteRequest(BaseModel):
    user_id: str = Field(..., alias="userId")
    destination_ids: Optional[List[str]] = Field(None, alias="destinationIds")
    province: Optional[str] = None
    start_destination_id: Optional[str] = Field(None, alias="startDestinationId")
    start_coordinates: Optional[CoordinateInput] = Field(None, alias="startCoordinates")
    start_label: Optional[str] = Field("Start", alias="startLabel")
    max_time: Optional[float] = Field(None, alias="maxTime")
    top_n: int = Field(5, alias="topN", ge=1, le=20)
    include_rated: bool = Field(False, alias="includeRated")
    random_seed: Optional[int] = Field(None, alias="randomSeed")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class RouteResponse(BaseModel):
    summary: Dict[str, Any]
    itinerary: Dict[str, Any]
    visualization: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True
