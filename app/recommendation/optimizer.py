from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

Coordinate = Tuple[float, float]


@dataclass
class RouteSolution:
    route: List[str]
    total_distance: float
    total_time: float
    metadata: Dict[str, object]


def build_graph(
    destinations: Iterable[Dict[str, object]],
    *,
    start_point: Optional[Dict[str, object]] = None,
    average_speed_kmh: float = 40.0,
) -> nx.Graph:
    """Create a complete weighted graph from destination coordinates."""
    graph = nx.Graph()

    if start_point is not None:
        _add_node(graph, start_point, is_start=True)

    for destination in destinations:
        _add_node(graph, destination, is_start=False)

    nodes = list(graph.nodes)
    for idx, source in enumerate(nodes):
        for target in nodes[idx + 1 :]:
            src_coord = graph.nodes[source]["coords"]
            tgt_coord = graph.nodes[target]["coords"]
            distance = _haversine_distance(src_coord, tgt_coord)
            time_hours = distance / max(average_speed_kmh, 1e-3)
            graph.add_edge(source, target, distance=distance, time=time_hours)

    return graph


def find_shortest_path(
    graph: nx.Graph,
    start: str,
    end: str,
    *,
    algorithm: str = "dijkstra",
    weight: str = "distance",
) -> RouteSolution:
    """Compute the shortest path between two nodes using Dijkstra or A*."""
    if algorithm.lower() == "astar":
        heuristic = _make_heuristic(graph, weight=weight)
        path = nx.astar_path(graph, start, end, heuristic=heuristic, weight=weight)
    else:
        path = nx.shortest_path(graph, start, end, weight=weight, method="dijkstra")

    total_distance, total_time = _measure_route(graph, path)
    metadata = {"algorithm": algorithm, "weight": weight}
    return RouteSolution(route=path, total_distance=total_distance, total_time=total_time, metadata=metadata)


def optimize_route(
    graph: nx.Graph,
    start: str,
    *,
    waypoints: Optional[Sequence[str]] = None,
    max_time: Optional[float] = None,
    population_size: int = 80,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism: int = 2,
    random_seed: Optional[int] = None,
) -> RouteSolution:
    """Use a genetic algorithm to find an efficient multi-stop itinerary."""
    if random_seed is not None:
        random.seed(random_seed)

    if waypoints is None:
        waypoints = [node for node in graph.nodes if node != start]
    else:
        waypoints = [node for node in waypoints if node in graph and node != start]

    if not waypoints:
        return RouteSolution(route=[start], total_distance=0.0, total_time=0.0, metadata={"reason": "no_waypoints"})

    population = [_random_route(waypoints) for _ in range(population_size)]
    best_route: List[str] = []
    best_score = float("inf")
    best_time = float("inf")

    fitness_history = []
    for _ in range(max(1, generations)):
        scored_population = []
        for route in population:
            total_distance, total_time = _score_route(graph, start, route, max_time)
            scored_population.append((route, total_distance, total_time))
            if total_distance < best_score:
                best_route, best_score, best_time = route, total_distance, total_time

        fitness_history.append(best_score)
        scored_population.sort(key=lambda item: item[1])
        next_generation = [entry[0] for entry in scored_population[:elitism]]

        while len(next_generation) < population_size:
            parent_a = _tournament_selection(scored_population)
            parent_b = _tournament_selection(scored_population)
            if random.random() < crossover_rate:
                child_a, child_b = _ordered_crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            if random.random() < mutation_rate:
                _mutate(child_a)
            if random.random() < mutation_rate:
                _mutate(child_b)

            next_generation.extend([child_a, child_b])

        population = next_generation[:population_size]

    route_sequence = [start, *best_route]
    total_distance, total_time = _measure_route(graph, route_sequence)
    metadata = {
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "max_time": max_time,
        "fitness_history": fitness_history,
    }
    return RouteSolution(route=route_sequence, total_distance=total_distance, total_time=total_time, metadata=metadata)


def visualize_route(graph: nx.Graph, solution: RouteSolution) -> Dict[str, object]:
    """Return a lightweight structure for visualizing the itinerary externally."""
    nodes_payload: List[Dict[str, object]] = []
    for node in solution.route:
        data = graph.nodes[node]
        nodes_payload.append({"id": node, "coords": data.get("coords"), "label": data.get("label", node)})

    edges_payload: List[Dict[str, object]] = []
    for current, nxt in zip(solution.route[:-1], solution.route[1:]):
        edge_data = graph.get_edge_data(current, nxt, default={})
        edges_payload.append(
            {
                "from": current,
                "to": nxt,
                "distance": float(edge_data.get("distance", 0.0)),
                "time": float(edge_data.get("time", 0.0)),
            }
        )

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "total_distance": float(solution.total_distance),
        "total_time": float(solution.total_time),
        "metadata": solution.metadata,
    }


def _add_node(graph: nx.Graph, descriptor: Dict[str, object], *, is_start: bool) -> None:
    if "id" in descriptor:
        node_id = str(descriptor["id"])
    elif "destinationId" in descriptor:
        node_id = str(descriptor["destinationId"])
    else:
        raise ValueError("Node descriptor requires 'id' or 'destinationId'.")

    coords = descriptor.get("coords") or descriptor.get("coordinates")
    if coords is None:
        raise ValueError(f"Node '{node_id}' is missing 'coords'.")
    if len(coords) != 2:
        raise ValueError(f"Coordinates for node '{node_id}' must be a pair.")

    coords_tuple = (float(coords[0]), float(coords[1]))
    label = descriptor.get("label") or descriptor.get("name") or node_id

    graph.add_node(
        node_id,
        coords=coords_tuple,
        label=label,
        is_start=is_start,
        data=descriptor,
    )


def _haversine_distance(coord_a: Coordinate, coord_b: Coordinate) -> float:
    radius = 6371.0
    lat_a, lon_a = np.radians(coord_a)
    lat_b, lon_b = np.radians(coord_b)

    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a

    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(radius * c)


def _make_heuristic(graph: nx.Graph, *, weight: str) -> callable:
    def heuristic(node: str, goal: str) -> float:
        coord_a = graph.nodes[node]["coords"]
        coord_b = graph.nodes[goal]["coords"]
        distance = _haversine_distance(coord_a, coord_b)
        if weight == "time":
            default_speed = 40.0
            for _, _, data in graph.edges(data=True):
                dist = data.get("distance", 0.0)
                time = data.get("time", 0.0)
                if time > 0:
                    default_speed = dist / time
                    break
            return distance / max(default_speed, 1e-3)
        return distance

    return heuristic


def _measure_route(graph: nx.Graph, sequence: Sequence[str]) -> Tuple[float, float]:
    total_distance = 0.0
    total_time = 0.0
    for current, nxt in zip(sequence[:-1], sequence[1:]):
        data = graph.get_edge_data(current, nxt)
        if data is None:
            raise ValueError(f"Missing edge between {current} and {nxt}.")
        total_distance += float(data.get("distance", 0.0))
        total_time += float(data.get("time", 0.0))
    return total_distance, total_time


def _score_route(
    graph: nx.Graph,
    start: str,
    route: Sequence[str],
    max_time: Optional[float],
) -> Tuple[float, float]:
    sequence = [start, *route]
    total_distance, total_time = _measure_route(graph, sequence)
    if max_time is not None and total_time > max_time:
        penalty = 1.0 + (total_time - max_time) / max(max_time, 1e-3)
        total_distance *= penalty
    return total_distance, total_time


def _random_route(waypoints: Sequence[str]) -> List[str]:
    route = list(waypoints)
    random.shuffle(route)
    return route


def _tournament_selection(pool: List[Tuple[List[str], float, float]], k: int = 3) -> List[str]:
    size = min(k, len(pool))
    contestants = random.sample(pool, k=size)
    contestants.sort(key=lambda item: item[1])
    return contestants[0][0][:]


def _ordered_crossover(parent_a: Sequence[str], parent_b: Sequence[str]) -> Tuple[List[str], List[str]]:
    size = len(parent_a)
    if size < 2:
        return list(parent_a), list(parent_b)

    start, end = sorted(random.sample(range(size), 2))
    child_a = parent_a[start:end]
    child_b = parent_b[start:end]

    def fill(child: List[str], parent: Sequence[str]) -> List[str]:
        result = list(child)
        for gene in parent:
            if gene not in result:
                result.append(gene)
        return result

    offspring_a = fill(list(child_a), parent_b[start:] + parent_b[:start])
    offspring_b = fill(list(child_b), parent_a[start:] + parent_a[:start])
    return offspring_a[:size], offspring_b[:size]


def _mutate(route: List[str]) -> None:
    if len(route) < 2:
        return
    idx_a, idx_b = random.sample(range(len(route)), 2)
    route[idx_a], route[idx_b] = route[idx_b], route[idx_a]
