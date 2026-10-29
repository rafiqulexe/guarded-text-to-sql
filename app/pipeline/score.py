"""Confidence scoring (SCORING, STATE.md §4). Weights come from config (SCORE_WEIGHT_*)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.config import Settings


@dataclass(frozen=True)
class ScoreInputs:
    verdict: Literal["yes", "partial", "no"]
    expects_rows: bool
    consistency: float  # agreeing / N
    repairs_used: int
    max_repairs: int
    row_count: int
    limit: int  # outer LIMIT of the executed SQL
    limit_applied: bool  # that LIMIT was injected or clamped by us, not written by the model


@dataclass(frozen=True)
class ScoreResult:
    score: float  # 0..1
    breakdown: dict[str, float]  # {judge, consistency, repairs, sanity}


SIGNALS = ("judge", "consistency", "repairs", "sanity")
JUDGE_VALUE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def weights_from(settings: "Settings") -> dict[str, float]:
    return {
        "judge": settings.score_weight_judge,
        "consistency": settings.score_weight_consistency,
        "repairs": settings.score_weight_repairs,
        "sanity": settings.score_weight_sanity,
    }


def sanity(inputs: ScoreInputs) -> float:
    """0 if empty but rows were expected; 0.5 if the row count equals a LIMIT *we* injected or
    clamped (the model's own LIMIT, e.g. "top 5", is never penalised); else 1."""
    if inputs.row_count == 0 and inputs.expects_rows:
        return 0.0
    if inputs.limit_applied and inputs.row_count == inputs.limit:
        return 0.5
    return 1.0


def score(inputs: ScoreInputs, weights: dict[str, float]) -> ScoreResult:
    repairs = 1.0 if inputs.max_repairs == 0 else 1 - inputs.repairs_used / inputs.max_repairs
    breakdown = {
        "judge": JUDGE_VALUE[inputs.verdict],
        "consistency": inputs.consistency,
        "repairs": max(0.0, repairs),
        "sanity": sanity(inputs),
    }
    total = sum(weights[name] * breakdown[name] for name in SIGNALS)
    return ScoreResult(min(1.0, max(0.0, total)), breakdown)


def make_scorer(settings: "Settings") -> Callable[[ScoreInputs], ScoreResult]:
    weights = weights_from(settings)
    return lambda inputs: score(inputs, weights)
