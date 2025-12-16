"""Compatibility shim for legacy imports.

The moderation configuration classes now live in ``app.moderation.config``.
Importing from ``app.config`` continues to work for existing code.
"""

from .moderation.config import AppConfig, InferenceConfig, PathConfig, TrainingConfig

__all__ = ["AppConfig", "InferenceConfig", "PathConfig", "TrainingConfig"]
