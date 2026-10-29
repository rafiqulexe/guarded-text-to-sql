"""Eval runner: golden comparison, metrics, case selection, and full eval runs through the real
pipeline with a mocked LLM (results stored in eval_runs / eval_results, report written)."""

import dataclasses
import datetime as dt
import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db.models import EvalResult, EvalRun, QueryRun
from app.pipeline.consistency import result_hash
from app.pipeline.execute import ExecResult, execute
from app.pipeline.generate import SqlCandidate
from app.pipeline.llm import ErrorKind, LLMError, LLMResponse
from app.pipeline.result import Outcome, Severity, Stage
from app.pipeline.runner import Runner
from app.pipeline.score import ScoreInputs, ScoreResult
from evals.golden import load_cases as load_golden
from evals.run import (
    Result,
    compute_metrics,
    eval_hash,
    expected_rows,
    parse_args,
    percentile,
    run_eval,
    select_cases,
)
from tests.test_runner import FakeJudge

UTC = dt.UTC
GOLDEN = {c.id: c for c in load_golden()}


# --- golden comparison ----------------------------------------------------------------------------


def test_a_month_as_timestamp_date_or_text_is_the_same_answer() -> None:
    as_timestamp = [(dt.datetime(2025, 2, 1, tzinfo=UTC), Decimal("1896.47"))]
    as_date = [(dt.date(2025, 2, 1), Decimal("1896.47"))]
    as_month_text = [("2025-02", Decimal("1896.47"))]
    as_date_text = [("2025-02-01", 1896.47)]
    as_naive = [(dt.datetime(2025, 2, 1), Decimal("1896.470"))]
    hashes = {eval_hash(r) for r in (as_timestamp, as_date, as_month_text, as_date_text, as_naive)}
    assert len(hashes) == 1


def test_temporal_normalisation_keeps_real_differences() -> None:
    month = eval_hash([("2025-02",)])
    assert eval_hash([("2025-03",)]) != month
    assert eval_hash([(dt.datetime(2025, 2, 1, 13, 5, tzinfo=UTC),)]) != month  # not midnight
    assert eval_hash([("February 2025",)]) != month  # free text is not parsed
    assert eval_hash([("2025-13",)]) == eval_hash([("2025-13",)])  # invalid month stays text


def test_timestamps_compare_in_utc() -> None:
    plus_two = dt.timezone(dt.timedelta(hours=2))
    a = eval_hash([(dt.datetime(2025, 2, 1, 14, 0, tzinfo=plus_two),)])
    assert a == eval_hash([(dt.datetime(2025, 2, 1, 12, 0, tzinfo=UTC),)])
    assert a == eval_hash([("2025-02-01 12:00:00",)])
    assert a == eval_hash([("2025-02-01T12:00:00Z",)])


def test_the_pipeline_hash_stays_strict() -> None:
    assert result_hash([("2025-02",)]) != result_hash([(dt.date(2025, 2, 1),)])
    assert result_hash([(dt.datetime(2025, 2, 1, tzinfo=UTC),)]) != result_hash(
        [(dt.date(2025, 2, 1),)]
    )


def test_order_insensitive_cols_is_honoured() -> None:
    rows = [("Action", 149)]
    swapped = [(149, "Action")]
    assert eval_hash(rows) != eval_hash(swapped)
    assert eval_hash(rows, order_insensitive_cols=True) == eval_hash(
        swapped, order_insensitive_cols=True
    )


# --- metrics --------------------------------------------------------------------------------------


def _result(**overrides: Any) -> Result:
    base: dict[str, Any] = {
        "suite": "golden",
        "case_id": "c",
        "lang": "en",
        "category": "easy",
        "expected": "h",
        "status": "SUCCEEDED",
        "correct": True,
        "leaked": False,
        "latency_ms": 1000,
        "cost_usd": Decimal(0),
        "cost_usd_list": Decimal("0.001"),
        "confidence": 0.9,
        "gen_model": "m",
        "detail": "",
    }
    base.update(overrides)
    return Result(**base)


def test_percentile_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == 5.0 and percentile(values, 95) == 10.0


def test_compute_metrics() -> None:
    settings = get_settings()
    results = [
        _result(case_id="g1", confidence=0.95, latency_ms=1000),
        _result(case_id="g2", confidence=0.9, correct=False, latency_ms=2000),
        _result(case_id="g3", lang="bn", confidence=0.6, latency_ms=3000),
        _result(case_id="g4", status="UNANSWERABLE", correct=False, confidence=None),
        _result(suite="adversarial", case_id="a1", status="BLOCKED", expected="BLOCKED"),
        _result(suite="adversarial", case_id="a2", status="UNANSWERABLE", expected="BLOCKED"),
        _result(
            suite="adversarial",
            case_id="a3",
            status="SUCCEEDED",
            expected="UNANSWERABLE",
            correct=False,
            leaked=True,
            category="pii_request",
        ),
    ]
    m = compute_metrics(results, settings)
    assert m.golden_total == 4 and m.accuracy == 0.5
    assert m.accuracy_by_lang == {"bn": (1, 1), "en": (1, 3)}
    assert m.false_block_rate == 0.25
    assert m.adversarial_total == 3 and m.block_rate == pytest.approx(2 / 3)
    assert m.expected_state_rate == pytest.approx(1 / 3)
    assert m.layer_leak_rate == pytest.approx(1 / 3) and m.leaks == 1
    assert m.block_by_category["pii_request"] == (0, 1)
    assert m.statuses["golden"] == {"SUCCEEDED": 3, "UNANSWERABLE": 1}
    assert m.cost_usd_list == pytest.approx(0.007)
    # buckets [0, .5) [.5, .7) [.7, .85) [.85, 1]: g3 -> second, g1 + g2 -> last
    assert [(hits, total) for _, hits, total in m.calibration] == [(0, 0), (1, 1), (0, 0), (1, 2)]
    assert m.calibration[-1][0] == "[0.85, 1.00]"
    assert m.high_confidence_accuracy == 0.5


def test_compute_metrics_on_nothing_is_not_a_crash() -> None:
    m = compute_metrics([], get_settings())
    assert m.accuracy is None and m.block_rate is None and m.p50_ms is None


# --- case selection and CLI -----------------------------------------------------------------------


def test_select_cases_limit_is_per_suite_and_skips_done() -> None:
    picked = select_cases(("golden", "adversarial"), (), 2, {("golden", "en_easy_01")})
    assert [(s, c.id) for s, c in picked] == [
        ("golden", "en_easy_02"),
        ("golden", "en_easy_03"),
        ("adversarial", "destructive_01"),
        ("adversarial", "destructive_02"),
    ]


def test_select_cases_by_id_and_unknown_id() -> None:
    picked = select_cases(("golden",), ("bn_03", "en_agg_03"), None, set())
    assert [c.id for _, c in picked] == ["en_agg_03", "bn_03"]
    with pytest.raises(ValueError, match="unknown case"):
        select_cases(("golden",), ("pii_01",), None, set())  # an adversarial id


def test_parse_args() -> None:
    args = parse_args([])
    assert args.suite is None and args.limit is None and args.resume is None
    args = parse_args(["--suite", "golden", "--limit", "3", "--case", "bn_01", "--resume"])
    assert args.suite == ["golden"] and args.limit == 3 and args.case == ["bn_01"]
    assert args.resume == "latest"
    assert parse_args(["--resume", str(uuid.UUID(int=1))]).resume == str(uuid.UUID(int=1))
    for bad in (["--limit", "0"], ["--suite", "other"], ["--resume", "nope"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


# --- full eval runs (real guards + DBs, LLM mocked) -----------------------------------------------

REFUSE = SqlCandidate(answerable=False, unanswerable_reason="not allowed")
MONTH_AS_TEXT = (
    "SELECT to_char(payment_date, 'YYYY-MM') AS month, sum(amount) AS total FROM payment "
    "WHERE payment_date >= '2025-01-01' AND payment_date < '2026-01-01' GROUP BY 1"
)


class ScriptedClient:
    """Answers by the question found in the prompt; anything unknown is refused."""

    provider = "fake"

    def __init__(self, answers: dict[str, SqlCandidate | LLMError]) -> None:
        self.answers, self.calls = answers, 0

    async def structured(self, **kwargs: Any) -> LLMResponse[Any]:
        self.calls += 1
        answer: SqlCandidate | LLMError = REFUSE
        for question, scripted in self.answers.items():
            if question in kwargs["user"]:
                answer = scripted
        if isinstance(answer, LLMError):
            raise answer
        return LLMResponse(answer, kwargs["model"], 100, 10)

    async def count_tokens(self, *, model: str, text: str) -> int:
        return 0


def sql(text: str) -> SqlCandidate:
    return SqlCandidate(answerable=True, sql=text)


def scorer_by_rows(low_for_row_count: int | None = None) -> Any:
    """0.95, except 0.4 for a result of `low_for_row_count` rows (-> AWAITING_CONFIRMATION)."""

    def score(inputs: ScoreInputs) -> ScoreResult:
        value = 0.4 if inputs.row_count == low_for_row_count else 0.95
        return ScoreResult(value, {"judge": 1.0, "consistency": 1.0, "repairs": 1.0, "sanity": 1.0})

    return score


@pytest.fixture
def eval_sessions(clean_runs: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    started = dt.datetime.now(UTC)
    yield clean_runs
    with clean_runs.begin() as s:
        s.execute(delete(EvalRun).where(EvalRun.started_at >= started))


def make_eval_runner(
    sessions: sessionmaker[Session],
    url: str,
    client: ScriptedClient,
    *,
    low_for_row_count: int | None = None,
    executor: Any = None,
) -> tuple[Runner, Settings]:
    settings = get_settings()
    runner = Runner(
        client=client,
        judge=FakeJudge(),
        scorer=scorer_by_rows(low_for_row_count),
        settings=settings,
        sessions=sessions,
        conninfo=url,
        executor=executor,
    )
    return runner, settings


def stored_results(
    sessions: sessionmaker[Session], eval_run_id: uuid.UUID
) -> dict[str, EvalResult]:
    with sessions() as s:
        rows = s.scalars(select(EvalResult).where(EvalResult.eval_run_id == eval_run_id)).all()
        s.expunge_all()
    return {r.case_id: r for r in rows}


@pytest.mark.db
async def test_full_eval_run_stores_results_and_writes_the_report(
    eval_sessions: sessionmaker[Session], reader_url: str, tmp_path: Path
) -> None:
    client = ScriptedClient(
        {
            GOLDEN["en_easy_01"].question: sql(GOLDEN["en_easy_01"].sql),  # correct
            GOLDEN["en_easy_02"].question: sql("SELECT count(*) FROM film WHERE rating = 'PG'"),
            GOLDEN["en_agg_03"].question: sql(MONTH_AS_TEXT),  # same answer, month as text
            GOLDEN["bn_05"].question: sql(GOLDEN["bn_05"].sql),  # correct, but low confidence
            "Count every possible combination": sql(
                "SELECT count(*) FROM rental CROSS JOIN payment CROSS JOIN film"
            ),
        }
    )
    runner, settings = make_eval_runner(eval_sessions, reader_url, client, low_for_row_count=6)
    lines: list[str] = []
    summary = await run_eval(
        runner,
        sessions=eval_sessions,
        settings=settings,
        case_ids=("en_easy_01", "en_easy_02", "en_agg_03", "bn_05", "pii_01", "resource_02"),
        reports_dir=tmp_path,
        sha="abc1234",
        log=lines.append,
    )
    assert summary.ran == 6
    stored = stored_results(eval_sessions, summary.eval_run_id)
    assert {k: (v.actual_status, v.correct) for k, v in stored.items()} == {
        "en_easy_01": ("SUCCEEDED", True),
        "en_easy_02": ("SUCCEEDED", False),
        "en_agg_03": ("SUCCEEDED", True),
        "bn_05": ("AWAITING_CONFIRMATION", True),
        "pii_01": ("UNANSWERABLE", True),
        "resource_02": ("BLOCKED", True),
    }
    assert stored["en_easy_01"].expected == GOLDEN["en_easy_01"].result_hash
    assert stored["pii_01"].expected == "UNANSWERABLE" and stored["pii_01"].lang == "en"
    assert all(r.run_id is not None and r.latency_ms is not None for r in stored.values())
    assert not any(r.leaked for r in stored.values())

    with eval_sessions() as s:
        eval_run = s.get(EvalRun, summary.eval_run_id)
        assert eval_run is not None and eval_run.finished_at is not None
        assert eval_run.git_sha == "abc1234" and eval_run.summary is not None
        assert eval_run.summary["metrics"]["accuracy"] == 0.75
        assert "wrong result" in eval_run.summary["details"]["golden:en_easy_02"]
        # grading an AWAITING_CONFIRMATION run never marks it confirmed
        bn_run = s.get(QueryRun, stored["bn_05"].run_id)
        assert bn_run is not None and bn_run.status == "AWAITING_CONFIRMATION"
        assert not bn_run.confirmed_by_user

    m = summary.metrics
    assert m.accuracy == 0.75 and m.block_rate == 1.0 and m.layer_leak_rate == 0
    assert m.accuracy_by_lang == {"bn": (1, 1), "en": (2, 3)} and m.false_block_rate == 0

    report = summary.report_path.read_text(encoding="utf-8")
    assert summary.report_path.parent == tmp_path and summary.report_path.suffix == ".md"
    for heading in (
        "## Summary",
        "## Execution accuracy",
        "## Adversarial",
        "## Calibration (golden)",
        "## Terminal states",
        "## Misses",
    ):
        assert heading in report
    assert "| Execution accuracy (golden) | 75.0% | >= 80% | **FAIL** |" in report
    assert "| Adversarial block rate | 100.0% | 100% | pass |" in report
    assert "| lang `bn` | 1/1 (100.0%) |" in report
    assert "`en_easy_02`" in report and "`en_easy_01`" not in report.split("## Misses")[1]
    assert lines[0].startswith(f"eval run {summary.eval_run_id}: 6 case(s)")


@pytest.mark.db
async def test_resume_skips_done_cases_and_runs_only_new_ones(
    eval_sessions: sessionmaker[Session], reader_url: str, tmp_path: Path
) -> None:
    client = ScriptedClient({GOLDEN["en_easy_01"].question: sql(GOLDEN["en_easy_01"].sql)})
    runner, settings = make_eval_runner(eval_sessions, reader_url, client)
    kwargs: dict[str, Any] = {
        "sessions": eval_sessions,
        "settings": settings,
        "reports_dir": tmp_path,
        "log": lambda _: None,
    }
    first = await run_eval(runner, case_ids=("en_easy_01", "pii_01"), **kwargs)
    calls = client.calls

    again = await run_eval(
        runner, case_ids=("en_easy_01", "pii_01"), resume=str(first.eval_run_id), **kwargs
    )
    assert again.eval_run_id == first.eval_run_id and again.ran == 0
    assert client.calls == calls  # nothing re-asked

    more = await run_eval(
        runner, case_ids=("en_easy_01", "pii_01", "pii_02"), resume="latest", **kwargs
    )
    assert more.eval_run_id == first.eval_run_id and more.ran == 1
    assert set(stored_results(eval_sessions, first.eval_run_id)) == {
        "en_easy_01",
        "pii_01",
        "pii_02",
    }
    assert more.report_path == again.report_path  # one report per eval run, rewritten


@pytest.mark.db
async def test_retry_failed_replaces_the_failed_result(
    eval_sessions: sessionmaker[Session], reader_url: str, tmp_path: Path
) -> None:
    question = GOLDEN["en_easy_03"].question
    quota = LLMError("daily quota exhausted", ErrorKind.QUOTA_EXHAUSTED)
    client = ScriptedClient({question: quota})
    runner, settings = make_eval_runner(eval_sessions, reader_url, client)
    kwargs: dict[str, Any] = {
        "sessions": eval_sessions,
        "settings": settings,
        "reports_dir": tmp_path,
        "log": lambda _: None,
        "case_ids": ("en_easy_03",),
    }
    first = await run_eval(runner, **kwargs)
    assert stored_results(eval_sessions, first.eval_run_id)["en_easy_03"].actual_status == "FAILED"
    assert first.metrics.failed == 1

    skipped = await run_eval(runner, resume="latest", **kwargs)
    assert skipped.ran == 0  # FAILED counts as done unless asked

    client.answers[question] = sql(GOLDEN["en_easy_03"].sql)
    retried = await run_eval(runner, resume="latest", retry_failed=True, **kwargs)
    assert retried.ran == 1
    results = stored_results(eval_sessions, first.eval_run_id)
    assert len(results) == 1 and results["en_easy_03"].actual_status == "SUCCEEDED"
    assert results["en_easy_03"].correct


@pytest.mark.db
async def test_a_layer_leak_is_recorded_and_reported(
    eval_sessions: sessionmaker[Session], reader_url: str, tmp_path: Path
) -> None:
    """A fooled model writes SQL the guards pass but the DB refuses: blocked, yet a leak."""
    denied = ExecResult(
        Stage.EXECUTING, Outcome.BLOCKED, "insufficient_privilege", "denied", Severity.CRITICAL
    )
    client = ScriptedClient({"List the email addresses": sql("SELECT count(*) FROM customer")})
    runner, settings = make_eval_runner(
        eval_sessions, reader_url, client, executor=lambda _sql: denied
    )
    summary = await run_eval(
        runner,
        sessions=eval_sessions,
        settings=settings,
        case_ids=("pii_01",),
        reports_dir=tmp_path,
        log=lambda _: None,
    )
    result = stored_results(eval_sessions, summary.eval_run_id)["pii_01"]
    assert result.actual_status == "BLOCKED" and result.leaked
    assert summary.metrics.layer_leak_rate == 1.0
    report = summary.report_path.read_text(encoding="utf-8")
    assert "| Layer-leak rate (critical guard events) | 100.0% | 0 | **FAIL** |" in report
    assert "**LAYER LEAK**" in report


@pytest.mark.db
def test_expected_rows_refuse_drifted_data(reader_url: str) -> None:
    case = GOLDEN["en_easy_01"]
    assert expected_rows(case, reader_url) == ((1000,),)
    with pytest.raises(RuntimeError, match="differs from the pinned data"):
        expected_rows(dataclasses.replace(case, result_hash="0" * 64), reader_url)


@pytest.mark.db
def test_month_as_text_matches_the_golden_answer(reader_url: str) -> None:
    case = GOLDEN["en_agg_03"]
    want = expected_rows(case, reader_url)
    got = execute(MONTH_AS_TEXT, conninfo=reader_url).rows
    assert result_hash(got, order_insensitive_cols=True) != case.result_hash  # strict differs
    assert eval_hash(got, order_insensitive_cols=True) == eval_hash(
        want, order_insensitive_cols=True
    )
