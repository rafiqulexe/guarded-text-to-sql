"""SCORING (STATE.md §4): signal values per the table, weights from config, sanity rule."""

from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.pipeline import ast_guard, rewrite, schema_guard
from app.pipeline.score import ScoreInputs, make_scorer, score, weights_from


def inputs(**overrides: Any) -> ScoreInputs:
    base: dict[str, Any] = {
        "verdict": "yes",
        "expects_rows": True,
        "consistency": 1.0,
        "repairs_used": 0,
        "max_repairs": 2,
        "row_count": 3,
        "limit": 100,
        "limit_applied": True,
    }
    return ScoreInputs(**(base | overrides))


WEIGHTS = weights_from(get_settings())


# --- signal values -------------------------------------------------------------------------------


@pytest.mark.parametrize(("verdict", "value"), [("yes", 1.0), ("partial", 0.5), ("no", 0.0)])
def test_judge_signal(verdict: str, value: float) -> None:
    assert score(inputs(verdict=verdict), WEIGHTS).breakdown["judge"] == value


@pytest.mark.parametrize(("used", "value"), [(0, 1.0), (1, 0.5), (2, 0.0)])
def test_repairs_signal_matches_the_table(used: int, value: float) -> None:
    assert score(inputs(repairs_used=used), WEIGHTS).breakdown["repairs"] == value


def test_repairs_signal_with_no_repairs_allowed() -> None:
    assert score(inputs(repairs_used=0, max_repairs=0), WEIGHTS).breakdown["repairs"] == 1.0


def test_consistency_signal_is_passed_through() -> None:
    assert score(inputs(consistency=0.5), WEIGHTS).breakdown["consistency"] == 0.5


@pytest.mark.parametrize(
    ("row_count", "expects_rows", "limit", "applied", "sanity"),
    [
        (0, True, 100, True, 0.0),  # empty but rows were expected
        (0, False, 100, True, 1.0),  # empty is a fine answer
        (100, True, 100, True, 0.5),  # hit a LIMIT we injected
        (1000, True, 1000, True, 0.5),  # hit a LIMIT we clamped
        (5, True, 5, False, 1.0),  # hit the model's own "top 5" LIMIT
        (42, True, 100, True, 1.0),  # under our LIMIT
    ],
)
def test_sanity_signal(
    row_count: int, expects_rows: bool, limit: int, applied: bool, sanity: float
) -> None:
    result = score(
        inputs(row_count=row_count, expects_rows=expects_rows, limit=limit, limit_applied=applied),
        WEIGHTS,
    )
    assert result.breakdown["sanity"] == sanity


def test_breakdown_keys_match_schema_md() -> None:
    assert set(score(inputs(), WEIGHTS).breakdown) == {"judge", "consistency", "repairs", "sanity"}


def test_perfect_and_worst_scores() -> None:
    assert score(inputs(row_count=3), WEIGHTS).score == pytest.approx(1.0)
    worst = inputs(verdict="no", consistency=0.0, repairs_used=2, row_count=0, expects_rows=True)
    assert score(worst, WEIGHTS).score == pytest.approx(0.0)


def test_default_weighted_sum() -> None:
    # judge partial .5*.45 + consistency .5*.25 + repairs .5*.15 + sanity 1*.15
    result = score(inputs(verdict="partial", consistency=0.5, repairs_used=1), WEIGHTS)
    assert result.score == pytest.approx(0.225 + 0.125 + 0.075 + 0.15)


# --- weights come from config ---------------------------------------------------------------------


def test_default_weights_are_the_state_md_table() -> None:
    assert WEIGHTS == {"judge": 0.45, "consistency": 0.25, "repairs": 0.15, "sanity": 0.15}


def test_changing_config_weights_changes_the_score() -> None:
    judge_only = get_settings().model_copy(
        update={
            "score_weight_judge": 1.0,
            "score_weight_consistency": 0.0,
            "score_weight_repairs": 0.0,
            "score_weight_sanity": 0.0,
        }
    )
    scorer = make_scorer(judge_only)
    assert scorer(
        inputs(verdict="partial", consistency=0.0, repairs_used=2)
    ).score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "weights",
    [
        {"score_weight_judge": 0.5},  # sum 1.05
        {"score_weight_sanity": -0.15, "score_weight_judge": 0.75},  # negative
    ],
)
def test_invalid_weights_are_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="SCORE_WEIGHT"):
        Settings(_env_file=None, gemini_api_key="k", **weights)  # type: ignore[arg-type]


# --- PLAN 4.3 acceptance, through the real REWRITING step ---------------------------------------


def _limit_of(sql: str) -> rewrite.RewriteResult:
    ast = ast_guard.check(sql)
    assert ast.expression is not None
    schema = schema_guard.check(ast.expression)
    assert schema.expression is not None
    return rewrite.rewrite(schema.expression, default_limit=100, max_limit=1000)


def test_acceptance_model_written_top_5_is_not_penalised() -> None:
    rewritten = _limit_of("SELECT title FROM film ORDER BY length DESC LIMIT 5")
    assert (rewritten.limit, rewritten.limit_applied) == (5, False)
    result = score(
        inputs(row_count=5, limit=rewritten.limit, limit_applied=rewritten.limit_applied), WEIGHTS
    )
    assert result.breakdown["sanity"] == 1.0


def test_acceptance_hitting_the_injected_limit_100_gets_half() -> None:
    rewritten = _limit_of("SELECT title FROM film")
    assert (rewritten.limit, rewritten.limit_applied) == (100, True)
    result = score(
        inputs(row_count=100, limit=rewritten.limit, limit_applied=rewritten.limit_applied), WEIGHTS
    )
    assert result.breakdown["sanity"] == 0.5
