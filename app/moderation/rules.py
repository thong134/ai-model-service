from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .data_processing import basic_clean


@dataclass
class RuleResult:
    score: float
    triggers: List[str]


def _normalize_elongated(text: str) -> str:
    """Collapse repeated characters (e.g., 'cặccccc' -> 'cặc', 'lồnnnn' -> 'lồn')."""
    return re.sub(r'(.)\1{2,}', r'\1', text)


class RuleEngine:
    def __init__(self, profanity_terms: Iterable[str], suspicion_terms: Iterable[str]):
        self.profanity_patterns: List[re.Pattern[str]] = [
            re.compile(rf"{re.escape(term)}", re.IGNORECASE) for term in profanity_terms
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
        # Normalize elongated characters for profanity detection
        normalized = _normalize_elongated(cleaned.lower())
        
        triggers: List[str] = []
        penalty = 0.0

        # Check profanity on normalized text (catches 'cặccccc', 'lồnnnn', etc.)
        profanity_hits = self._matches(normalized, self.profanity_patterns)
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


# Comprehensive Vietnamese profanity list
DEFAULT_PROFANITY = [
    # Core vulgar terms
    "đm", "đmm", "đmcs", "đcm", "dcm", "dm", "dmm",
    "địt", "dit", "đjt", "djt",
    "đéo", "deo", "đ3o",
    "cặc", "cac", "kặc", "cak",
    "lồn", "lon", "l0n", "loz", "lòn",
    "buồi", "buoi", "bùi",
    "dái", "dai",
    "đĩ", "di", "đỉ",
    "cave",
    # Animal insults
    "chó", "cho", "ch0",
    "óc chó", "oc cho",
    "đồ chó", "do cho",
    "con chó", "con cho",
    "ngu", "ngu si", "ngu vl", "ngu vcl",
    "đần", "dan", "đần độn",
    "khùng", "khung",
    "điên", "dien",
    "thần kinh", "than kinh",
    # Insults
    "khốn nạn", "khon nan",
    "đồ khốn", "do khon",
    "mẹ mày", "me may",
    "má nó", "ma no",
    "bố mày", "bo may",
    "cha mày", "cha may",
    "con mẹ", "con me",
    "mày", 
    "thằng", "thang",
    "con điếm", "con diem",
    "đồ điếm", "do diem",
    "đồ rác", "do rac",
    "rác rưởi", "rac ruoi",
    "đồ phế", "do phe",
    "vô học", "vo hoc",
    "mất dạy", "mat day",
    "vô dụng", "vo dung",
    "đồ ngu", "do ngu",
    "thằng ngu", "thang ngu",
    "con ngu", "con ngu",
    # Abbreviations and slang
    "vcl", "vl", "vkl", "vcc", "cc",
    "clm", "cmm", "cmnr",
    "wtf", "wth",
    "đkm", "dkm",
    "đcmm", "dcmm",
    "clgt",
    "đmml",
    # English profanity
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitty",
    "bitch", "bitchy",
    "asshole", "ass",
    "dick", "dickhead",
    "pussy",
    "cunt",
    "bastard",
    "idiot", "idiots",
    "stupid",
    "loser",
    "suck", "sucks", "sux",
    "damn", "dammit",
    "hell",
    "wtf", "stfu",
    "trash", "garbage",
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
    "inbox",
    "liên hệ ngay",
    "nhắn tin",
    "zalo",
    "telegram",
]


def build_default_rule_engine() -> RuleEngine:
    return RuleEngine(DEFAULT_PROFANITY, DEFAULT_SUSPICION)
