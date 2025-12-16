from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import AppConfig
from .model import ModerationModel, ModerationPrediction
from .rules import RuleEngine, RuleResult, build_default_rule_engine


class ReviewService:
    def __init__(self, config: Optional[AppConfig] = None, rule_engine: Optional[RuleEngine] = None):
        self.config = config or AppConfig()
        self.rule_engine = rule_engine or build_default_rule_engine()
        self.model = ModerationModel(self.config)
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.model.load()
            self._loaded = True

    def _probability(self, prediction: ModerationPrediction, head: str, label: str) -> float:
        mapping = {
            "sentiment": prediction.sentiment,
            "toxicity": prediction.toxicity,
            "spam": prediction.spam,
        }
        target = mapping[head]
        return ModerationModel.probability_for_label(target, label)

    def _decide(self, toxicity: float, spam: float, rule_result: RuleResult) -> Tuple[str, List[str]]:
        cfg = self.config.inference
        reasons: List[str] = list(rule_result.triggers)
        decision = "approve"

        if toxicity >= cfg.toxicity_reject_threshold:
            decision = "reject"
            reasons.append("toxicity_high")
        elif spam >= cfg.spam_reject_threshold:
            decision = "reject"
            reasons.append("spam_high")
        elif rule_result.score >= cfg.rule_reject_threshold:
            decision = "reject"
            reasons.append("rule_reject")
        else:
            manual_triggers: List[str] = []
            if toxicity >= cfg.toxicity_manual_threshold:
                manual_triggers.append("toxicity_manual")
            if spam >= cfg.spam_manual_threshold:
                manual_triggers.append("spam_manual")
            if rule_result.score >= cfg.rule_manual_threshold and not rule_result.triggers:
                manual_triggers.append("rule_manual")

            if manual_triggers or rule_result.triggers:
                decision = "manual_review"
                reasons.extend(manual_triggers)

        deduped: List[str] = []
        for reason in reasons:
            if reason not in deduped:
                deduped.append(reason)

        return decision, deduped

    def score(self, text: str) -> Dict[str, object]:
        if not self._loaded:
            self.load()
        prediction = self.model.predict_one(text)
        rule_result = self.rule_engine.evaluate(text)

        toxicity_score = self._probability(prediction, "toxicity", "toxic")
        spam_score = self._probability(prediction, "spam", "spam")
        decision, reasons = self._decide(toxicity_score, spam_score, rule_result)

        return {
            "decision": decision,
            "reasons": reasons,
            "sentiment": {
                "label": prediction.sentiment.label,
                "confidence": float(prediction.sentiment.confidence),
                "scores": prediction.sentiment.probabilities,
            },
            "toxicity": {
                "label": prediction.toxicity.label,
                "score": float(toxicity_score),
                "confidence": float(prediction.toxicity.confidence),
                "scores": prediction.toxicity.probabilities,
            },
            "spam": {
                "label": prediction.spam.label,
                "score": float(spam_score),
                "confidence": float(prediction.spam.confidence),
                "scores": prediction.spam.probabilities,
            },
            "rules": {
                "score": float(rule_result.score),
                "triggers": rule_result.triggers,
            },
        }
