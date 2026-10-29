"""Token + cost accounting: per LLM call and per run, the model that actually answered
(fallback included), real cost (0 on the free tier) and the cost at paid list prices."""

from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import MODEL_PRICES_USD_PER_MTOK, get_settings
from app.pipeline.accounting import Ledger, list_cost
from app.pipeline.generate import SqlCandidate
from app.pipeline.llm import ErrorKind, LLMError, LLMResponse
from app.pipeline.runner import Runner
from app.pipeline.validate import JudgeVerdict
from tests.test_runner import GOOD, PII, UNKNOWN_COL, ok, stored

YES = JudgeVerdict(verdict="yes", expects_rows=True, reason="ok")


# --- prices ------------------------------------------------------------------------------------


def test_list_cost_per_million_tokens() -> None:
    assert list_cost("gemini-3.8-flash", 1_000_000, 1_000_000) == Decimal("4.500000")
    assert list_cost("gemini-3.5-flash-lite", 2000, 500) == Decimal("0.001850")  # .0006 + .00125
    assert list_cost("claude-haiku-5-5", 1_000_000, 0) == Decimal("0.100000")


def test_unknown_model_has_no_price() -> None:
    assert list_cost("some-new-model", 100, 100) is None


def test_price_table_covers_every_default_model() -> None:
    settings = get_settings()
    for model in (settings.gen_model, settings.judge_model, settings.llm_fallback_model):
        assert model in MODEL_PRICES_USD_PER_MTOK


# --- ledger ---------------------------------------------------------------------------------------


def _response(
    model: str, tokens_in: int, tokens_out: int, fallback_from: str | None = None
) -> LLMResponse[Any]:
    return LLMResponse(YES, model, tokens_in, tokens_out, 2, fallback_from)


def test_free_tier_real_cost_is_zero_but_list_cost_is_kept() -> None:
    ledger = Ledger("free")
    ledger.record(
        "generate",
        "gemini-3.8-flash",
        candidate=1,
        response=_response("gemini-3.8-flash", 1000, 100),
    )
    fields = ledger.fields()
    assert fields["cost_usd"] == Decimal(0)
    assert fields["cost_usd_list"] == Decimal("0.001125")  # 1000*.75e-6 + 100*3.75e-6
    assert (fields["input_tokens"], fields["output_tokens"]) == (1000, 100)


def test_paid_billing_real_cost_equals_list_cost() -> None:
    ledger = Ledger("paid")
    ledger.record(
        "judge", "gemini-3.5-flash-lite", response=_response("gemini-3.5-flash-lite", 2000, 500)
    )
    assert ledger.cost_usd == ledger.cost_usd_list == Decimal("0.001850")


def test_fallback_is_recorded_and_priced_at_the_answering_model() -> None:
    ledger = Ledger("free")
    ledger.record(
        "generate",
        "gemini-3.8-flash",
        candidate=1,
        response=_response("gemini-3.5-flash-lite", 2000, 500, fallback_from="gemini-3.8-flash"),
    )
    call = ledger.calls[0]
    assert (call["requested_model"], call["model"], call["fallback_from"]) == (
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    )
    assert call["cost_usd_list"] == "0.001850" and call["attempts"] == 2


def test_failed_call_is_recorded_with_zero_tokens() -> None:
    ledger = Ledger("free")
    ledger.record(
        "generate", "gemini-3.8-flash", candidate=2, error=LLMError("503", ErrorKind.RATE_LIMITED)
    )
    call = ledger.calls[0]
    assert call["model"] is None and call["input_tokens"] == 0 and "rate_limited" in call["error"]


def test_unpriced_model_is_flagged() -> None:
    ledger = Ledger("free")
    ledger.record("judge", "x", response=_response("mystery-model", 10, 10))
    assert ledger.calls[0]["price_known"] is False and ledger.cost_usd_list == Decimal(0)


# --- per run, through the runner (fake LLM, real DBs) --------------------------------------------


class ScriptedClient:
    """Returns scripted (answer, model, fallback_from) per call, in call order."""

    provider = "fake"

    def __init__(self, *script: tuple[Any, str, str | None] | LLMError) -> None:
        self.script = list(script)

    async def structured(self, **kwargs: Any) -> LLMResponse[Any]:
        item = self.script.pop(0)
        if isinstance(item, LLMError):
            raise item
        answer, model, fallback_from = item
        tokens = (300, 30) if isinstance(answer, JudgeVerdict) else (1500, 120)
        return LLMResponse(answer, model, *tokens, 1, fallback_from)

    async def count_tokens(self, *, model: str, text: str) -> int:
        return 0


def _runner(
    sessions: sessionmaker[Session], url: str, client: ScriptedClient, **overrides: Any
) -> Runner:
    settings = get_settings().model_copy(update=overrides)
    return Runner(client=client, settings=settings, sessions=sessions, conninfo=url)


GEN, LITE = "gemini-3.8-flash", "gemini-3.5-flash-lite"


def _cost(model: str, tokens_in: int, tokens_out: int) -> Decimal:
    cost = list_cost(model, tokens_in, tokens_out)
    assert cost is not None
    return cost


@pytest.mark.db
async def test_succeeded_run_records_every_call_and_both_costs(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    client = ScriptedClient((ok(GOOD), GEN, None), (ok(GOOD), LITE, GEN), (YES, LITE, None))
    outcome = await _runner(clean_runs, reader_url, client).ask("How many PG films?")
    assert outcome.status == "SUCCEEDED"
    row, _ = stored(clean_runs, outcome.run_id)
    calls = row.llm_calls
    assert [(c["purpose"], c["candidate"], c["model"], c["fallback_from"]) for c in calls] == [
        ("generate", 1, GEN, None),
        ("generate", 2, LITE, GEN),  # fell back: the answering model is recorded
        ("judge", None, LITE, None),
    ]
    assert (row.input_tokens, row.output_tokens) == (1500 * 2 + 300, 120 * 2 + 30)
    expected_list = _cost(GEN, 1500, 120) + _cost(LITE, 1500, 120) + _cost(LITE, 300, 30)
    assert row.cost_usd == Decimal(0)  # Gemini free tier
    assert row.cost_usd_list == expected_list
    assert (row.gen_model, row.judge_model) == (GEN, LITE)


@pytest.mark.db
async def test_paid_billing_stores_the_real_cost(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    client = ScriptedClient((ok(GOOD), GEN, None), (ok(GOOD), GEN, None), (YES, LITE, None))
    outcome = await _runner(clean_runs, reader_url, client, llm_billing="paid").ask("q")
    row, _ = stored(clean_runs, outcome.run_id)
    assert row.cost_usd == row.cost_usd_list > 0


@pytest.mark.db
async def test_primary_answered_by_fallback_sets_gen_model(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    client = ScriptedClient((ok(GOOD), LITE, GEN), (ok(GOOD), LITE, GEN), (YES, LITE, None))
    outcome = await _runner(clean_runs, reader_url, client).ask("q")
    row, _ = stored(clean_runs, outcome.run_id)
    assert row.gen_model == LITE  # what really wrote the executed SQL


@pytest.mark.db
async def test_blocked_run_keeps_its_accounting(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    client = ScriptedClient((ok(GOOD), GEN, None), (ok(PII), GEN, None))
    outcome = await _runner(clean_runs, reader_url, client).ask("q")
    assert outcome.status == "BLOCKED"
    row, _ = stored(clean_runs, outcome.run_id)
    assert len(row.llm_calls) == 2 and row.input_tokens == 3000 and row.cost_usd_list > 0


@pytest.mark.db
async def test_repair_and_failed_calls_are_recorded(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    down = LLMError("503 high demand", ErrorKind.RATE_LIMITED)
    client = ScriptedClient(
        (ok(UNKNOWN_COL), GEN, None), down, (ok(GOOD), GEN, None), (YES, LITE, None)
    )
    outcome = await _runner(clean_runs, reader_url, client).ask("q")
    assert outcome.status == "SUCCEEDED"
    row, _ = stored(clean_runs, outcome.run_id)
    assert [c["purpose"] for c in row.llm_calls] == ["generate", "generate", "repair", "judge"]
    assert row.llm_calls[1]["error"] and row.llm_calls[1]["input_tokens"] == 0


@pytest.mark.db
async def test_unanswerable_run_keeps_its_accounting(
    clean_runs: sessionmaker[Session], reader_url: str
) -> None:
    nope = SqlCandidate(answerable=False, unanswerable_reason="no weather data")
    client = ScriptedClient((nope, GEN, None), (nope, GEN, None))
    outcome = await _runner(clean_runs, reader_url, client).ask("weather?")
    assert outcome.status == "UNANSWERABLE"
    row, _ = stored(clean_runs, outcome.run_id)
    assert len(row.llm_calls) == 2 and row.output_tokens == 240
