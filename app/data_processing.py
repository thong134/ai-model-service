"""Compatibility shim for legacy imports.

Tokenization helpers were moved to ``app.moderation.data_processing``.
"""

from .moderation.data_processing import (
    basic_clean,
    normalize_repeated_characters,
    preprocess_for_tfidf,
    preprocess_for_vectorizer,
    strip_accents,
    tokenize,
)

__all__ = [
    "basic_clean",
    "normalize_repeated_characters",
    "preprocess_for_tfidf",
    "preprocess_for_vectorizer",
    "strip_accents",
    "tokenize",
]
