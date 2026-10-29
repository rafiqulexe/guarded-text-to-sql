"""Confidence scoring (SCORING, STATE.md §4)."""

from dataclasses import dataclass
from typing import Literal


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
