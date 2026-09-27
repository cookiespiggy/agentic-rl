from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierDecision:
    tier: str
    reason: str


def route_by_confidence(intent_confidence: float, complexity: float, escalation_prob: float, cheap_threshold: float, mid_threshold: float) -> TierDecision:
    if escalation_prob >= 0.7 or complexity >= 0.85:
        return TierDecision("premium", "high_risk")
    if intent_confidence >= cheap_threshold and escalation_prob < 0.4 and complexity < 0.5:
        return TierDecision("cheap", "high_conf_low_risk")
    if intent_confidence >= mid_threshold:
        return TierDecision("mid", "medium_conf")
    return TierDecision("premium", "low_conf_fallback")

