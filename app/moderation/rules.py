from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .data_processing import basic_clean


@dataclass
class RuleResult:
    score: float
    triggers: List[str]


class RuleEngine:
    def __init__(self, profanity_terms: Iterable[str], suspicion_terms: Iterable[str]):
        self.profanity_patterns: List[re.Pattern[str]] = [
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in profanity_terms
        ]
        self.suspicion_patterns: List[re.Pattern[str]] = [
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in suspicion_terms
        ]
        self.noise_pattern = re.compile(r"(.)\1{4,}")

    def _matches(self, text: str, patterns: Iterable[re.Pattern[str]]) -> List[str]:
        matched: List[str] = []
        for pattern in patterns:
            if pattern.search(text):
                matched.append(pattern.pattern)
        return matched

    def evaluate(self, text: str) -> RuleResult:
        cleaned = basic_clean(text)
        triggers: List[str] = []
        penalty = 0.0

        profanity_hits = self._matches(cleaned, self.profanity_patterns)
        if profanity_hits:
            triggers.extend(["profanity" for _ in profanity_hits])
            penalty += 1.0

        suspicion_hits = self._matches(cleaned, self.suspicion_patterns)
        if suspicion_hits:
            triggers.extend(["suspicious_term" for _ in suspicion_hits])
            penalty += 0.3

        if len(cleaned.split()) <= 3:
            triggers.append("too_short")
            penalty += 0.2

        if len(cleaned.split()) >= 100:
            triggers.append("too_long")
            penalty += 0.2

        if self.noise_pattern.search(text):
            triggers.append("noise_repetition")
            penalty += 0.4

        return RuleResult(score=min(penalty, 1.0), triggers=triggers)


DEFAULT_PROFANITY = [
    # Vietnamese
    "đm",
    "đmm",
    "địt",
    "đéo",
    "cặc",
    "lồn",
    "chó",
    "đĩ",
    "óc chó",
    "khốn nạn",
    "mẹ mày",
    "má nó",
    "ngu",
    "ngu si",
    "toxic",
    # English
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "dick",
    "pussy",
    "cunt",
    "bastard",
    "idiot",
    "stupid",
    "loser",
    "suck",
    "sux",
]

DEFAULT_SUSPICION = [
    "link",
    "http",
    "https",
    "click",
    "mua ngay",
    "giảm giá",
    "khuyến mãi",
    "free",
    "buy now",
    "click here",
    "cash prize",
    "win now",
]


def build_default_rule_engine() -> RuleEngine:
    return RuleEngine(DEFAULT_PROFANITY, DEFAULT_SUSPICION)
