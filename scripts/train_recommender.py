from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.recommender_system import load_model, recommend_for_user, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and persist the hybrid destination recommender.")
    parser.add_argument("--users", type=Path, default=Path("data/users.csv"), help="Path to users CSV file.")
    parser.add_argument(
        "--destinations",
        type=Path,
        default=Path("data/destinations.csv"),
        help="Path to destinations CSV file.",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("data/feedback.csv"),
        help="Path to feedback CSV file.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/recommender"),
        help="Directory where the trained model artifacts will be stored.",
    )
    parser.add_argument("--latent-factors", type=int, default=32, help="Number of latent factors for SVD.")
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Minimum rating considered positive when building user profiles.",
    )
    parser.add_argument(
        "--min-profile-items",
        type=int,
        default=3,
        help="Minimum number of items to use when building user profiles for sparse users.",
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("COLLAB", "CONTENT", "POPULARITY"),
        default=None,
        help="Optional custom weights for collaborative, content, and popularity components.",
    )
    parser.add_argument(
        "--preview-user",
        type=str,
        default=None,
        help="Optional userId to preview recommendations for after training.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to display when previewing results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for command output.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    weights = None
    if args.weights:
        weights = {
            "collaborative": max(args.weights[0], 0.0),
            "content": max(args.weights[1], 0.0),
            "popularity": max(args.weights[2], 0.0),
        }

    logging.info("Training hybrid recommender...")
    model = train_model(
        users_path=args.users,
        destinations_path=args.destinations,
        feedback_path=args.feedback,
        latent_factors=args.latent_factors,
        positive_feedback_threshold=args.positive_threshold,
        min_profile_items=args.min_profile_items,
        weights=weights,
    )

    artifact_path = save_model(model, args.artifacts)
    logging.info("Artifacts stored at %s", artifact_path)

    if args.preview_user:
        logging.info("Generating preview recommendations for user '%s'", args.preview_user)
        recommendations = recommend_for_user(model, args.preview_user, top_n=args.top_n)
        if not recommendations:
            logging.warning("No recommendations produced for the specified user.")
        else:
            print(json.dumps(recommendations, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
