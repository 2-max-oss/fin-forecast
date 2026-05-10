"""Composite scoring and verdict computation."""
from __future__ import annotations

from datetime import datetime

from config import DEFAULT_WEIGHTS, RATING_BANDS, CONFIDENCE_BAND_HALF_WIDTH
from core.types import CompositeResult, ModuleScore, Rating


def _score_to_rating(score: float) -> Rating:
    for low, high, label in RATING_BANDS:
        if low <= score <= high:
            return Rating(label)
    return Rating.HOLD


class CompositeScorer:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute(self, scores: list[ModuleScore]) -> CompositeResult:
        available = {s.name: s for s in scores if s is not None}

        # Redistribute weight from missing modules proportionally
        missing = [k for k in self.weights if k not in available]
        active_weight_sum = sum(v for k, v in self.weights.items() if k not in missing)

        effective_weights: dict[str, float] = {}
        if active_weight_sum > 0:
            for k, v in self.weights.items():
                if k in available:
                    effective_weights[k] = v / active_weight_sum
        else:
            # All modules missing — equal weight over whatever we have
            for name in available:
                effective_weights[name] = 1.0 / len(available) if available else 0.0

        # Weighted average score
        overall = sum(
            available[name].score * w
            for name, w in effective_weights.items()
            if name in available
        )

        # Confidence = weighted average of module confidences
        confidence = sum(
            available[name].confidence * w
            for name, w in effective_weights.items()
            if name in available
        ) if effective_weights else 0.0

        # Confidence band
        half_width = CONFIDENCE_BAND_HALF_WIDTH * (1.0 - confidence)
        band_low  = max(0.0, overall - half_width)
        band_high = min(100.0, overall + half_width)

        rating = _score_to_rating(overall)

        # Possible ratings if band spans multiple
        possible = []
        seen = set()
        for s in [band_low, overall, band_high]:
            r = _score_to_rating(s)
            if r not in seen:
                possible.append(r)
                seen.add(r)

        # Per-module weighted contributions
        component_scores = {
            name: available[name].score * effective_weights.get(name, 0.0)
            for name in available
        }

        return CompositeResult(
            overall_score=round(overall, 1),
            confidence=round(confidence, 3),
            rating=rating,
            confidence_band=(round(band_low, 1), round(band_high, 1)),
            possible_ratings=possible,
            weights_used=effective_weights,
            component_scores=component_scores,
            module_scores=list(available.values()),
            computed_at=datetime.utcnow(),
        )
