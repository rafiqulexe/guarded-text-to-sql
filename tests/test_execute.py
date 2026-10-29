"""EXECUTING (layer 4): READ ONLY transaction as sqlguard_reader, SET LOCAL statement_timeout,
fetch rows + columns, ROLLBACK. Runs against the docker Postgres."""

import time

import psycopg
import pytest

from app.allowlist import load_allowlist
from app.pipeline import ast_guard, execute, rewrite, schema_guard
from app.pipeline.result import Outcome, Severity, Stage

pytestmark = pytest.mark.db


def _guarded(sql: str) -> str:
    ast = ast_guard.check(sql)
    assert ast.passed and ast.expression is not None, ast.detail
    schema = schema_guard.check(ast.expression)
    assert schema.passed and schema.expression is not None, schema.detail
    return rewrite.rewrite(schema.expression, default_limit=100, max_limit=1000).sql


def _execute(sql: str, url: str, timeout_ms: int = 5000) -> execute.ExecResult:
    result = execute.execute(sql, conninfo=url, timeout_ms=timeout_ms, max_rows=1000)
    assert result.stage is Stage.EXECUTING
    return result


# --- happy path ------------------------------------------------------------------


def test_returns_columns_and_rows(reader_url: str) -> None:
    result = _execute(
        _guarded("SELECT title, rating FROM film ORDER BY film_id LIMIT 3"), reader_url
    )
    assert (result.outcome, result.rule) == (Outcome.PASSED, "executed"), result.detail
    assert result.columns == ("title", "rating")
    assert result.row_count == 3
    assert result.rows[0] == ("ACADEMY DINOSAUR", "PG")


def test_select_star_customer_runs_without_touching_email(reader_url: str) -> None:
    """The expanded * never reaches the DB permission layer on email."""
    result = _execute(_guarded("SELECT * FROM customer"), reader_url)
    assert result.passed, result.detail
    assert result.columns == tuple(load_allowlist().tables["customer"].columns)
    assert result.row_count == 100  # injected DEFAULT_LIMIT


def test_runs_in_a_read_only_transaction_with_local_timeout(reader_url: str) -> None:
    result = _execute(
        "SELECT current_setting('transaction_read_only'), current_setting('statement_timeout')",
        reader_url,
        timeout_ms=1234,
    )
    assert result.rows == (("on", "1234ms"),)


def test_max_rows_caps_the_fetch(reader_url: str) -> None:
    result = execute.execute("SELECT film_id FROM film", conninfo=reader_url, max_rows=10)
    assert result.row_count == 10


def test_defaults_come_from_settings(reader_url: str) -> None:
    result = execute.execute(_guarded("SELECT count(*) FROM film"))
    assert result.rows == ((1000,),)


# --- timeout -> FAILED (not repaired) ------------------------------


def test_timeout_is_failed(reader_url: str) -> None:
    # A real heavy query (no banned functions): count over a 2.6e9-row cross join.
    started = time.monotonic()
    result = _execute("SELECT count(*) FROM rental r, payment p", reader_url, timeout_ms=300)
    assert (result.outcome, result.rule) == (Outcome.FAILED, "timeout"), result.detail
    assert time.monotonic() - started < 5


# --- DB errors -> repair --------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    ["SELECT nope FROM film", "SELECT * FROM movies", "SELECT 1/0", "SELEC 1"],
)
def test_db_errors_go_to_repair(reader_url: str, sql: str) -> None:
    result = _execute(sql, reader_url)
    assert (result.outcome, result.rule) == (Outcome.REPAIR, "db_error"), result.detail
    assert result.rows == ()


# --- a guard leak reaching the DB is a critical BLOCK, and nothing changes ------------


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("SELECT email FROM customer", "insufficient_privilege"),
        ("SELECT pg_sleep(5)", "insufficient_privilege"),
        ("SELECT film_in_stock(1, 1)", "insufficient_privilege"),
        ("DELETE FROM film", "read_only_violation"),
        ("UPDATE film SET title = 'x'", "read_only_violation"),
        ("SELECT 1; DELETE FROM film", "multiple_statements"),
    ],
)
def test_guard_leak_is_blocked_critical(reader_url: str, sql: str, rule: str) -> None:
    result = _execute(sql, reader_url)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, rule), result.detail
    assert result.severity is Severity.CRITICAL
    assert result.rows == ()


def test_nothing_was_modified(reader_url: str) -> None:
    for sql in ("DELETE FROM film", "SELECT 1; DELETE FROM film"):
        _execute(sql, reader_url)
    with psycopg.connect(reader_url) as conn:
        row = conn.execute("SELECT count(*) FROM film").fetchone()
    assert row == (1000,)


# --- no DB needed -----------------------------------------------------------------------


def test_unreachable_db_is_failed() -> None:
    url = "postgresql://sqlguard_reader:reader_pw@127.0.0.1:1/pagila"
    result = execute.execute("SELECT 1", conninfo=url, timeout_ms=1000, max_rows=10)
    assert (result.outcome, result.rule) == (Outcome.FAILED, "db_unavailable")
