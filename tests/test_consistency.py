"""Self-consistency with the real guards against the docker Postgres and a fake
LLM client (no key). Some cases also fake the executor to reach execution-time outcomes."""

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest

from app.pipeline import consistency
from app.pipeline.consistency import result_hash, run_self_consistency
from app.pipeline.execute import ExecResult
from app.pipeline.generate import SqlCandidate
from app.pipeline.llm import ErrorKind, LLMError, LLMResponse
from app.pipeline.result import Outcome, Severity, Stage, blocked, failed, repair

REF = dt.date(2026, 7, 29)
GOOD = "SELECT count(*) FROM film WHERE rating = 'PG'"
GOOD_SAME = "SELECT count(film_id) FROM film WHERE rating = 'PG'"  # same result, other SQL
GOOD_OTHER = "SELECT count(*) FROM film WHERE rating = 'G'"
UNKNOWN_COL = "SELECT count(*) FROM film WHERE certificate = 'PG'"
PII = "SELECT email FROM customer"


def ok(sql: str) -> SqlCandidate:
    return SqlCandidate(answerable=True, sql=sql, tables_used=["film"])


NOPE = SqlCandidate(answerable=False, unanswerable_reason="not in this database")


class FakeClient:
    """Answers in call order; candidates are started in order, so answer i = candidate i+1."""

    provider = "fake"

    def __init__(self, *answers: SqlCandidate | LLMError) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    async def structured(self, **kwargs: Any) -> LLMResponse[Any]:
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, LLMError):
            raise answer
        return LLMResponse(answer, kwargs["model"], 100, 10)

    async def count_tokens(self, *, model: str, text: str) -> int:
        return 0


async def _run(
    client: FakeClient, url: str, n: int = 2, **kwargs: Any
) -> consistency.ConsistencyResult:
    return await run_self_consistency(
        "How many PG films are there?",
        client=client,
        model="gemini-3.8-flash",
        reference_date=REF,
        n=n,
        max_repairs=kwargs.pop("max_repairs", 2),
        conninfo=url,
        **kwargs,
    )


# --- result hash -------------------------------------------------------------------


def test_hash_ignores_row_order_and_column_names() -> None:
    assert result_hash([(1, "a"), (2, "b")]) == result_hash([(2, "b"), (1, "a")])


def test_hash_normalises_numbers_to_two_decimals() -> None:
    assert result_hash([(Decimal("3.00"),)]) == result_hash([(3,)]) == result_hash([(3.0,)])
    assert result_hash([(Decimal("1.234"),)]) == result_hash([(1.23,)])
    assert result_hash([(Decimal("1.24"),)]) != result_hash([(1.23,)])


def test_hash_normalises_timestamps_to_iso_utc() -> None:
    utc = dt.datetime(2026, 7, 1, 12, 0, tzinfo=dt.UTC)
    plus2 = dt.datetime(2026, 7, 1, 14, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert result_hash([(utc,)]) == result_hash([(plus2,)])
    assert result_hash([(dt.date(2026, 7, 1),)]) != result_hash([(utc,)])


def test_hash_distinguishes_null_bool_and_text() -> None:
    hashes = {result_hash([(v,)]) for v in (None, "None", True, 1, "true", "")}
    assert len(hashes) == 6


def test_column_order_matters_unless_insensitive() -> None:
    assert result_hash([(1, "a")]) != result_hash([("a", 1)])
    assert result_hash([(1, "a")], order_insensitive_cols=True) == result_hash(
        [("a", 1)], order_insensitive_cols=True
    )


def test_hash_handles_arrays() -> None:
    assert result_hash([(["Trailers", "Commentaries"],)]) != result_hash([(["Trailers"],)])


# --- agreement ------------------------------------------------------------------------------------


@pytest.mark.db
async def test_agreeing_candidates_give_full_consistency(reader_url: str) -> None:
    client = FakeClient(ok(GOOD), ok(GOOD_SAME))
    result = await _run(client, reader_url)
    assert result.next_state == "VALIDATING"
    assert result.consistency == 1.0
    assert result.primary is not None and result.primary.index == 1
    assert [c["temperature"] for c in client.calls] == [0.0, 0.4]
    assert result.primary.result_hash == result.candidates[1].result_hash
    assert result.columns and result.rows == ((194,),)


@pytest.mark.db
async def test_disagreeing_candidates_halve_consistency(reader_url: str) -> None:
    result = await _run(FakeClient(ok(GOOD), ok(GOOD_OTHER)), reader_url)
    assert (result.next_state, result.consistency) == ("VALIDATING", 0.5)


@pytest.mark.db
async def test_n_of_one(reader_url: str) -> None:
    result = await _run(FakeClient(ok(GOOD)), reader_url, n=1)
    assert (result.next_state, result.consistency) == ("VALIDATING", 1.0)


@pytest.mark.db
async def test_injected_limit_is_recorded_for_the_primary(reader_url: str) -> None:
    result = await _run(
        FakeClient(ok("SELECT title FROM film"), ok("SELECT title FROM film")), reader_url
    )
    assert result.primary is not None
    assert result.primary.limit_applied and result.primary.final_sql.endswith("LIMIT 100")
    assert len(result.rows) == 100


# --- any BLOCKED candidate blocks the run (safety beats availability) -----------------------------


@pytest.mark.db
@pytest.mark.parametrize(
    "bad",
    [
        PII,
        "SELECT 1; DROP TABLE film",
        "SELECT pg_sleep(10)",
        "SELECT * FROM pg_shadow",
        "SELECT r.rental_id FROM rental r, payment p, film f",
    ],
)
async def test_one_blocked_candidate_blocks_the_run_and_nothing_executes(
    reader_url: str, bad: str
) -> None:
    result = await _run(FakeClient(ok(GOOD), ok(bad)), reader_url)
    assert result.next_state == "BLOCKED"
    assert result.blocking is not None and result.blocking.outcome is Outcome.BLOCKED
    assert all(c.result_hash is None for c in result.candidates)  # no candidate was executed
    assert result.rows == ()


@pytest.mark.db
async def test_blocked_candidate_is_never_repaired(reader_url: str) -> None:
    client = FakeClient(ok(PII), ok(PII), ok(GOOD))
    result = await _run(client, reader_url)
    assert result.next_state == "BLOCKED" and result.repairs_used == 0
    assert len(client.calls) == 2  # no repair call was made


# --- answerable: candidate #1 decides -------------------------------------------------------------


@pytest.mark.db
async def test_candidate_one_unanswerable_means_unanswerable(reader_url: str) -> None:
    result = await _run(FakeClient(NOPE, ok(GOOD)), reader_url)
    assert result.next_state == "UNANSWERABLE"
    assert all(c.result_hash is None for c in result.candidates)


@pytest.mark.db
async def test_later_unanswerable_candidate_counts_as_non_agreeing(reader_url: str) -> None:
    result = await _run(FakeClient(ok(GOOD), NOPE), reader_url)
    assert (result.next_state, result.consistency) == ("VALIDATING", 0.5)
    assert result.candidates[1].dropped == "unanswerable"


# --- repairable candidates are dropped; repair only when all fail ---------------------------------


@pytest.mark.db
async def test_repairable_candidate_is_dropped_not_repaired(reader_url: str) -> None:
    client = FakeClient(ok(UNKNOWN_COL), ok(GOOD))
    result = await _run(client, reader_url)
    assert result.next_state == "VALIDATING" and result.repairs_used == 0
    assert result.primary is not None and result.primary.index == 2  # first survivor
    assert result.candidates[0].dropped == "repair"
    assert result.consistency == 0.5
    assert len(client.calls) == 2


@pytest.mark.db
async def test_all_fail_then_candidate_one_is_repaired(reader_url: str) -> None:
    client = FakeClient(ok(UNKNOWN_COL), ok("SELECT nope FROM film"), ok(GOOD))
    result = await _run(client, reader_url)
    assert (result.next_state, result.repairs_used) == ("VALIDATING", 1)
    repair_call = client.calls[2]
    assert "certificate" in repair_call["user"] and "<previous_sql>" in repair_call["user"]
    assert UNKNOWN_COL in repair_call["user"]  # repairs apply to candidate #1
    assert result.consistency == 0.5  # 1 survivor (the repaired primary) / N


@pytest.mark.db
async def test_repair_returning_blocked_sql_blocks_the_run(reader_url: str) -> None:
    client = FakeClient(ok(UNKNOWN_COL), ok(UNKNOWN_COL), ok(PII))
    result = await _run(client, reader_url)
    assert result.next_state == "BLOCKED" and result.repairs_used == 1


@pytest.mark.db
async def test_repairs_exhausted_is_failed(reader_url: str) -> None:
    client = FakeClient(ok(UNKNOWN_COL), ok(UNKNOWN_COL), ok(UNKNOWN_COL), ok(UNKNOWN_COL))
    result = await _run(client, reader_url, max_repairs=2)
    assert (result.next_state, result.repairs_used) == ("FAILED", 2)
    assert len(client.calls) == 4


@pytest.mark.db
async def test_answerable_without_sql_is_repairable(reader_url: str) -> None:
    empty = SqlCandidate(answerable=True, sql=None)
    result = await _run(FakeClient(empty, empty, ok(GOOD)), reader_url)
    assert (result.next_state, result.repairs_used) == ("VALIDATING", 1)


# --- LLM errors -----------------------------------------------------------------------------------

OVERLOADED = LLMError("503", ErrorKind.RATE_LIMITED)


@pytest.mark.db
async def test_failed_llm_call_drops_that_candidate(reader_url: str) -> None:
    result = await _run(FakeClient(OVERLOADED, ok(GOOD)), reader_url)
    assert result.next_state == "VALIDATING"
    assert result.candidates[0].dropped == "llm_error"
    assert result.primary is not None and result.primary.index == 2


@pytest.mark.db
async def test_all_llm_calls_failing_is_failed(reader_url: str) -> None:
    result = await _run(FakeClient(OVERLOADED, OVERLOADED), reader_url)
    assert result.next_state == "FAILED"


# --- execution-time outcomes (fake executor) ------------------------------------------------------


def _executor(by_sql: dict[str, ExecResult]) -> Any:
    def run(sql: str) -> ExecResult:
        for needle, outcome in by_sql.items():
            if needle in sql:
                return outcome
        return ExecResult(
            Stage.EXECUTING,
            Outcome.PASSED,
            "executed",
            "1 rows",
            Severity.INFO,
            columns=("n",),
            rows=((1,),),
        )

    return run


def _as_exec(result: Any) -> ExecResult:
    return ExecResult(result.stage, result.outcome, result.rule, result.detail, result.severity)


@pytest.mark.db
async def test_permission_error_at_execution_blocks_the_run(reader_url: str) -> None:
    leak = _as_exec(
        blocked(Stage.EXECUTING, "insufficient_privilege", "permission denied", critical=True)
    )
    result = await _run(
        FakeClient(ok(GOOD), ok(GOOD_OTHER)), reader_url, executor=_executor({"'G'": leak})
    )
    assert result.next_state == "BLOCKED"
    assert result.blocking is not None and result.blocking.severity is Severity.CRITICAL


@pytest.mark.db
async def test_primary_timeout_is_failed(reader_url: str) -> None:
    timeout = _as_exec(failed(Stage.EXECUTING, "timeout", "statement_timeout exceeded"))
    result = await _run(
        FakeClient(ok(GOOD), ok(GOOD_OTHER)), reader_url, executor=_executor({"'PG'": timeout})
    )
    assert result.next_state == "FAILED"


@pytest.mark.db
async def test_non_primary_timeout_is_dropped(reader_url: str) -> None:
    timeout = _as_exec(failed(Stage.EXECUTING, "timeout", "statement_timeout exceeded"))
    result = await _run(
        FakeClient(ok(GOOD), ok(GOOD_OTHER)), reader_url, executor=_executor({"'G'": timeout})
    )
    assert (result.next_state, result.consistency) == ("VALIDATING", 0.5)
    assert result.candidates[1].dropped == "execution_failed"


@pytest.mark.db
async def test_primary_db_error_at_execution_goes_to_repair(reader_url: str) -> None:
    db_error = _as_exec(repair(Stage.EXECUTING, "db_error", "division by zero"))
    fixed = "SELECT count(*) FROM film WHERE rating = 'R'"
    client = FakeClient(ok(GOOD), ok(GOOD_OTHER), ok(fixed))
    result = await _run(client, reader_url, executor=_executor({"'PG'": db_error}))
    assert (result.next_state, result.repairs_used) == ("VALIDATING", 1)
    assert "division by zero" in client.calls[2]["user"]  # exact DB error fed back
    assert result.primary is not None and "'R'" in result.primary.final_sql


# --- logging shape (query_runs.candidates) ----------------------------------------------


@pytest.mark.db
async def test_candidates_log_entries(reader_url: str) -> None:
    result = await _run(FakeClient(ok(GOOD), ok(UNKNOWN_COL)), reader_url)
    entries = result.candidates_log()
    assert [set(e) >= {"sql", "result_hash", "error"} for e in entries] == [True, True]
    assert entries[0]["result_hash"] and entries[0]["error"] is None
    assert entries[1]["result_hash"] is None and "certificate" in entries[1]["error"]
    assert result.input_tokens == 200 and result.output_tokens == 20
