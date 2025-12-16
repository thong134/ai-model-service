from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.travel_service import TravelService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an optimized travel route for a user.")
    parser.add_argument("user_id", type=str, help="User identifier to personalize the itinerary.")
    parser.add_argument(
        "--province",
        type=str,
        default=None,
        help="Optional province filter to keep destinations within a single region.",
    )
    parser.add_argument(
        "--start-destination",
        type=str,
        default=None,
        help="Destination ID to use as the starting point (must exist in destinations.csv).",
    )
    parser.add_argument(
        "--start-lat",
        type=float,
        default=None,
        help="Latitude for a custom starting location.",
    )
    parser.add_argument(
        "--start-lon",
        type=float,
        default=None,
        help="Longitude for a custom starting location.",
    )
    parser.add_argument(
        "--start-label",
        type=str,
        default="Start",
        help="Label for the custom starting location.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional maximum travel time in hours for the itinerary.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of destinations to include in the optimized route.",
    )
    parser.add_argument(
        "--destinations",
        nargs="+",
        default=None,
        help="Optional list of destination IDs to prioritize when building the route.",
    )
    parser.add_argument(
        "--include-rated",
        action="store_true",
        help="Allow already-rated destinations to appear in recommendations.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/recommender"),
        help="Directory containing the recommender artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing destination CSV data.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=40.0,
        help="Average travel speed in km/h when estimating travel time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to make the genetic algorithm deterministic.",
    )
    return parser.parse_args()


def resolve_start(args: argparse.Namespace) -> dict | str:
    if args.start_destination:
        return args.start_destination
    if args.start_lat is not None and args.start_lon is not None:
        return {"id": "__start__", "label": args.start_label, "coords": (args.start_lat, args.start_lon)}
    raise ValueError("Provide either --start-destination or both --start-lat and --start-lon.")


def main() -> None:
    args = parse_args()
    start_descriptor = resolve_start(args)

    service = TravelService(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts,
        average_speed_kmh=args.speed,
    )

    try:
        plan = service.plan_route(
            user_id=args.user_id,
            province=args.province,
            start=start_descriptor,
            destination_ids=args.destinations,
            max_time_hours=args.max_time,
            top_n=args.top_n,
            include_rated=args.include_rated,
            random_seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output = {
        "summary": plan.summary,
        "itinerary": plan.itinerary,
        "visualization": plan.visualization,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
