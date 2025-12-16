"""Compatibility shim for legacy moderation imports.

The moderation models now live in ``app.moderation``. This module keeps the old
``app.model`` import path working while the rest of the codebase migrates.
"""

from __future__ import annotations

from .moderation.legacy import DatasetSplits, ReviewModel
from .moderation.model import HeadPrediction, ModerationModel, ModerationPrediction, ModelHead

__all__ = [
    "DatasetSplits",
    "HeadPrediction",
    "ModerationModel",
    "ModerationPrediction",
    "ModelHead",
    "ReviewModel",
]
