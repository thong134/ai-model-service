"""Compatibility shim for legacy imports.

The review moderation service now lives in ``app.moderation.service``.
"""

from .moderation.service import ReviewService

__all__ = ["ReviewService"]
