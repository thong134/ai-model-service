from __future__ import annotations

import argparse
import uvicorn

from app.http_api import create_fastapi_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI server for travel recommendations.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--origins",
        nargs="*",
        default=None,
        help="Optional list of allowed CORS origins (defaults include localhost:3000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_fastapi_app(allowed_origins=args.origins)
    uvicorn.run(app, host=args.host, port=args.port)


app = create_fastapi_app()


if __name__ == "__main__":
    main()
