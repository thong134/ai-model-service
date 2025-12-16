"""Compatibility shim for legacy route optimizer imports.

Route optimisation helpers now live in ``app.itinerary.route_optimizer``. This
module keeps the historic ``app.route_optimizer`` path functional during the
migration.
"""

from __future__ import annotations

from .itinerary.route_optimizer import (
    RouteSolution,
    build_graph,
    find_shortest_path,
    optimize_route,
    visualize_route,
)

__all__ = [
    "RouteSolution",
    "build_graph",
    "find_shortest_path",
    "optimize_route",
    "visualize_route",
]
