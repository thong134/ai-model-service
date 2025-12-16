from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..recommender import HybridRecommenderModel, load_model, recommend_for_user
from .route_optimizer import RouteSolution, build_graph, optimize_route, visualize_route


@dataclass
class RoutePlan:
    summary: Dict[str, object]
    itinerary: Dict[str, object]
    visualization: Dict[str, object]


class TravelService:
    """High-level orchestration for destination recommendations and itinerary planning."""

    def __init__(
        self,
        *,
        data_dir: str | Path = Path("data"),
        artifacts_dir: str | Path = Path("artifacts/recommender"),
        average_speed_kmh: float = 40.0,
        recommender_model: Optional[HybridRecommenderModel] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.average_speed_kmh = float(average_speed_kmh)
        self._recommender_model = recommender_model
        self.destinations = self._load_destination_data()
        self.destination_lookup = self.destinations.set_index("destinationId")

    def load_recommender(self) -> HybridRecommenderModel:
        if self._recommender_model is None:
            self._recommender_model = load_model(self.artifacts_dir)
        return self._recommender_model

    def plan_route(
        self,
        user_id: str,
        *,
        province: Optional[str],
        start: str | Dict[str, object] | Sequence[float],
        destination_ids: Optional[Sequence[str]] = None,
        max_time_hours: Optional[float] = None,
        top_n: int = 5,
        include_rated: bool = False,
        random_seed: Optional[int] = None,
    ) -> RoutePlan:
        model = self.load_recommender()
        recommendations = recommend_for_user(
            model,
            user_id,
            top_n=max(top_n * 3, top_n),
            include_rated=include_rated,
        )

        explicit_ids = self._resolve_destination_ids(destination_ids, province, top_n)
        selected_ids = explicit_ids if explicit_ids else self._select_destinations(recommendations, province, top_n)
        if not selected_ids:
            raise ValueError("No destinations available for the requested itinerary.")

        start_descriptor = self._normalize_start(start)
        start_id = start_descriptor.get("id") or start_descriptor.get("destinationId")
        if not start_id:
            raise ValueError("Start point must provide an identifier via 'id' or 'destinationId'.")
        start_descriptor.setdefault("id", start_id)

        if start_id in selected_ids:
            selected_ids = [dest_id for dest_id in selected_ids if dest_id != start_id]
        if not selected_ids:
            raise ValueError("Selected destinations only include the start location. Provide additional stops.")

        destination_nodes = [self._destination_node(dest_id) for dest_id in selected_ids]
        graph = build_graph(destination_nodes, start_point=start_descriptor, average_speed_kmh=self.average_speed_kmh)
        waypoints = [node["destinationId"] for node in destination_nodes]

        solution = optimize_route(
            graph,
            start=start_descriptor["id"],
            waypoints=waypoints,
            max_time=max_time_hours,
            random_seed=random_seed,
        )

        itinerary = self._build_itinerary(graph, solution)
        visualization = visualize_route(graph, solution)
        summary = {
            "userId": user_id,
            "province": province,
            "total_distance_km": float(solution.total_distance),
            "total_time_hours": float(solution.total_time),
            "num_stops": len(solution.route) - 1,
            "optimizer": solution.metadata,
        }
        return RoutePlan(summary=summary, itinerary=itinerary, visualization=visualization)

    def recommend_destinations(
        self,
        user_id: str,
        *,
        top_n: int = 10,
        include_rated: bool = False,
        province: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        model = self.load_recommender()
        raw = recommend_for_user(model, user_id, top_n=max(top_n, 1), include_rated=include_rated)
        if province:
            normalized = province.lower()
            raw = [rec for rec in raw if rec.get("province", "").lower() == normalized]

        trimmed = raw[:top_n]
        sanitized: List[Dict[str, object]] = []
        for entry in trimmed:
            payload = dict(entry)
            rating = payload.get("averageRating")
            if rating is not None and (not isinstance(rating, (int, float)) or not math.isfinite(float(rating))):
                payload["averageRating"] = None
            sanitized.append(payload)
        return sanitized

    def _load_destination_data(self) -> pd.DataFrame:
        destinations_path = self.data_dir / "destinations.csv"
        coords_path = self.data_dir / "destination_coordinates.csv"
        if not destinations_path.exists():
            raise FileNotFoundError(f"Missing destinations dataset at '{destinations_path}'.")
        if not coords_path.exists():
            raise FileNotFoundError(f"Missing coordinates dataset at '{coords_path}'.")

        destinations = pd.read_csv(destinations_path)
        coords = pd.read_csv(coords_path)
        merged = destinations.merge(coords, on="destinationId", how="inner")
        merged["destinationId"] = merged["destinationId"].astype(str)
        merged["province"] = merged["province"].astype(str)
        merged = merged.dropna(subset=["latitude", "longitude"])
        merged["latitude"] = merged["latitude"].astype(float)
        merged["longitude"] = merged["longitude"].astype(float)
        merged["averageRating"] = pd.to_numeric(merged.get("averageRating"), errors="coerce")
        merged["averageRating"] = merged["averageRating"].fillna(merged["averageRating"].mean())
        if "name" not in merged.columns:
            merged["name"] = merged["category"].astype(str).str.title()
        return merged

    def _resolve_destination_ids(
        self,
        destination_ids: Optional[Sequence[str]],
        province: Optional[str],
        top_n: int,
    ) -> List[str]:
        if not destination_ids:
            return []

        normalized_province = province.lower() if isinstance(province, str) else None
        resolved: List[str] = []

        for dest in destination_ids:
            dest_id = str(dest)
            if dest_id not in self.destination_lookup.index:
                raise ValueError(f"Unknown destination '{dest_id}'.")
            if dest_id in resolved:
                continue
            if normalized_province is not None:
                province_value = str(self.destination_lookup.loc[dest_id]["province"]).lower()
                if province_value != normalized_province:
                    continue
            resolved.append(dest_id)
            if len(resolved) >= top_n:
                return resolved

        if normalized_province is not None and not resolved:
            for dest in destination_ids:
                dest_id = str(dest)
                if dest_id not in self.destination_lookup.index:
                    continue
                if dest_id in resolved:
                    continue
                resolved.append(dest_id)
                if len(resolved) >= top_n:
                    break

        return resolved

    def _select_destinations(
        self,
        recommendations: List[Dict[str, object]],
        province: Optional[str],
        top_n: int,
    ) -> List[str]:
        def add_if_valid(dest_id: str) -> None:
            if dest_id in seen:
                return
            if dest_id not in self.destination_lookup.index:
                return
            seen.add(dest_id)
            selected.append(dest_id)

        normalized_province = province.lower() if isinstance(province, str) else None
        selected: List[str] = []
        seen: set[str] = set()

        filtered = [rec for rec in recommendations if normalized_province is None or rec.get("province", "").lower() == normalized_province]
        for bucket in (filtered, recommendations):
            for rec in bucket:
                dest_id = str(rec.get("destinationId"))
                add_if_valid(dest_id)
                if len(selected) >= top_n:
                    return selected

        fallback = self.destinations
        if normalized_province is not None:
            fallback = fallback[fallback["province"].str.lower() == normalized_province]
        fallback = fallback.sort_values("averageRating", ascending=False, na_position="last")
        for dest_id in fallback["destinationId"].tolist():
            add_if_valid(str(dest_id))
            if len(selected) >= top_n:
                break

        return selected

    def _destination_node(self, dest_id: str) -> Dict[str, object]:
        if dest_id not in self.destination_lookup.index:
            raise ValueError(f"Unknown destination '{dest_id}'.")
        row = self.destination_lookup.loc[dest_id]
        return {
            "destinationId": str(dest_id),
            "coords": (float(row["latitude"]), float(row["longitude"])),
            "label": str(row.get("name", dest_id)),
            "category": row.get("category"),
            "province": row.get("province"),
            "description": row.get("description", ""),
            "averageRating": float(row.get("averageRating", 0.0)),
        }

    def _normalize_start(self, start: str | Dict[str, object] | Sequence[float]) -> Dict[str, object]:
        if isinstance(start, str):
            node = self._destination_node(start)
            node["id"] = node["destinationId"]
            node["label"] = node.get("label") or node["destinationId"]
            return node

        if isinstance(start, (list, tuple)) and len(start) == 2:
            return {"id": "__start__", "label": "Start", "coords": (float(start[0]), float(start[1]))}

        if isinstance(start, dict):
            descriptor = dict(start)
            node_id = descriptor.get("id") or descriptor.get("destinationId") or "__start__"
            coords = descriptor.get("coords") or descriptor.get("coordinates")
            if coords is None or len(coords) != 2:
                raise ValueError("Start descriptor requires 'coords' with two numeric values.")
            descriptor["id"] = str(node_id)
            descriptor["coords"] = (float(coords[0]), float(coords[1]))
            descriptor.setdefault("label", descriptor.get("name") or descriptor["id"])
            return descriptor

        raise TypeError("Start point must be a destinationId string, coordinate pair, or descriptor dictionary.")

    def _build_itinerary(self, graph, solution: RouteSolution) -> Dict[str, object]:
        stops: List[Dict[str, object]] = []
        for node_id in solution.route:
            data = graph.nodes[node_id]
            descriptor = data.get("data", {})
            stop_payload = {
                "id": node_id,
                "label": data.get("label", node_id),
                "coords": data.get("coords"),
                "is_start": bool(data.get("is_start")),
            }
            if descriptor:
                stop_payload.update(
                    {
                        "destinationId": descriptor.get("destinationId"),
                        "category": descriptor.get("category"),
                        "province": descriptor.get("province"),
                        "description": descriptor.get("description"),
                        "averageRating": descriptor.get("averageRating"),
                    }
                )
            stops.append(stop_payload)

        legs: List[Dict[str, object]] = []
        for current, nxt in zip(solution.route[:-1], solution.route[1:]):
            edge_data = graph.get_edge_data(current, nxt, default={})
            legs.append(
                {
                    "from": current,
                    "to": nxt,
                    "distance": float(edge_data.get("distance", 0.0)),
                    "time": float(edge_data.get("time", 0.0)),
                }
            )

        return {"stops": stops, "legs": legs}
