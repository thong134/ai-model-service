"""Compatibility shim for legacy imports.

The travel orchestration logic now resides in ``app.itinerary.travel_service``.
"""

from .itinerary.travel_service import RoutePlan, TravelService

__all__ = ["RoutePlan", "TravelService"]