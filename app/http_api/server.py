from __future__ import annotations

import os
from typing import Iterable, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import recommendations_router, route_router


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def create_fastapi_app(*, allowed_origins: Optional[Iterable[str]] = None) -> FastAPI:
    app = FastAPI(title="Traveline AI Recommendation API", version="1.0.0")

    origins = list(allowed_origins) if allowed_origins else _origins_from_env()
    if not origins:
        origins = DEFAULT_ALLOWED_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(recommendations_router)
    app.include_router(route_router)

    return app


def _origins_from_env() -> List[str]:
    raw = os.getenv("FASTAPI_ALLOWED_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
