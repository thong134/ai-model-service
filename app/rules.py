def build_default_rule_engine() -> RuleEngine:
"""Compatibility shim for legacy imports.

The rule engine now lives in ``app.moderation.rules``.
"""

from .moderation.rules import (
    DEFAULT_PROFANITY,
    DEFAULT_SUSPICION,
    RuleEngine,
    RuleResult,
    build_default_rule_engine,
)

__all__ = [
    "DEFAULT_PROFANITY",
    "DEFAULT_SUSPICION",
    "RuleEngine",
    "RuleResult",
    "build_default_rule_engine",
]
