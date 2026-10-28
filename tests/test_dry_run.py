"""DRY_RUN (layer 3): EXPLAIN (FORMAT JSON) as sqlguard_reader; BLOCKED if any plan node
exceeds MAX_COST or MAX_PLAN_ROWS. Runs against the docker Postgres."""

import pytest

from app.pipeline import ast_guard, dry_run, rewrite, schema_guard
from app.pipeline.result import Outcome, Severity, Stage

pytestmark = pytest.mark.db

MAX_COST, MAX_ROWS = 100_000, 1_000_000


def _guarded(sql: str) -> str:
    """The SQL DRY_RUN really sees: AST_CHECK -> SCHEMA_CHECK -> REWRITING (LIMIT injected)."""
    ast = ast_guard.check(sql)
    assert ast.passed and ast.expression is not None, ast.detail
    schema = schema_guard.check(ast.expression)
    assert schema.passed and schema.expression is not None, schema.detail
    return rewrite.rewrite(schema.expression, default_limit=100, max_limit=1000).sql


def _dry_run(
    sql: str, url: str, *, max_cost: float = MAX_COST, max_plan_rows: float = MAX_ROWS
) -> dry_run.DryRunResult:
    result = dry_run.dry_run(sql, conninfo=url, max_cost=max_cost, max_plan_rows=max_plan_rows)
    assert result.stage is Stage.DRY_RUN
    return result


# --- passes ---------------------------------------------------------------------

SIMPLE = [
    "SELECT title FROM film WHERE rating = 'PG'",
    "SELECT * FROM customer",
    "SELECT c.name, count(*) AS rentals FROM rental r"
    " JOIN inventory i ON i.inventory_id = r.inventory_id"
    " JOIN film_category fc ON fc.film_id = i.film_id"
    " JOIN category c ON c.category_id = fc.category_id GROUP BY c.name ORDER BY rentals DESC",
    "SELECT customer_id, sum(amount) FROM payment GROUP BY customer_id ORDER BY 2 DESC",
]


@pytest.mark.parametrize("sql", SIMPLE)
def test_simple_query_passes(reader_url: str, sql: str) -> None:
    result = _dry_run(_guarded(sql), reader_url)
    assert (result.outcome, result.rule) == (Outcome.PASSED, "plan_ok"), result.detail
    assert 0 < result.max_cost <= MAX_COST
    assert 0 < result.max_plan_rows <= MAX_ROWS


# --- BLOCKED: too expensive, even when the injected LIMIT hides it at the top node -----

CROSS_JOINS = [
    "SELECT r.rental_id, p.amount, f.title FROM rental r, payment p, film f",
    "SELECT r.rental_id FROM rental r CROSS JOIN payment p CROSS JOIN film f",
    "SELECT count(*) FROM rental r, payment p, film f",
    "SELECT r.rental_id, p.payment_id FROM rental r, payment p",
    "SELECT f.title FROM film f JOIN payment p ON true JOIN rental r ON true ORDER BY f.title",
]


@pytest.mark.parametrize("sql", CROSS_JOINS)
def test_cross_join_blocked(reader_url: str, sql: str) -> None:
    guarded = _guarded(sql)
    assert "LIMIT 100" in guarded
    result = _dry_run(guarded, reader_url)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "max_cost"), result.detail
    assert result.max_cost > MAX_COST


def test_rows_threshold(reader_url: str) -> None:
    sql = _guarded("SELECT payment_id FROM payment ORDER BY amount")  # sorts all ~51k rows
    assert _dry_run(sql, reader_url).passed
    result = _dry_run(sql, reader_url, max_plan_rows=1000)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "max_plan_rows")


def test_cost_threshold(reader_url: str) -> None:
    sql = _guarded("SELECT title FROM film")
    result = _dry_run(sql, reader_url, max_cost=0.001)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "max_cost")


def test_thresholds_default_to_settings(reader_url: str) -> None:
    assert dry_run.dry_run(_guarded("SELECT title FROM film"), conninfo=reader_url).passed
    blocked = dry_run.dry_run(_guarded(CROSS_JOINS[0]), conninfo=reader_url)
    assert blocked.outcome is Outcome.BLOCKED


# --- DB errors (guards bypassed on purpose) ---------------------------------------


def test_unknown_column_goes_to_repair(reader_url: str) -> None:
    result = _dry_run("SELECT nope FROM film LIMIT 10", reader_url)
    assert (result.outcome, result.rule) == (Outcome.REPAIR, "db_error")
    assert "nope" in result.detail


def test_permission_error_is_a_critical_block(reader_url: str) -> None:
    """InsufficientPrivilege means layers 1-2 leaked (STATE.md §5)."""
    result = _dry_run("SELECT email FROM customer LIMIT 10", reader_url)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "insufficient_privilege")
    assert result.severity is Severity.CRITICAL


def test_second_statement_is_rejected_by_the_protocol(reader_url: str) -> None:
    result = _dry_run("SELECT 1; SELECT 2", reader_url)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "multiple_statements")
    assert result.severity is Severity.CRITICAL


def test_unreachable_db_is_failed() -> None:
    url = "postgresql://sqlguard_reader:reader_pw@127.0.0.1:1/pagila?connect_timeout=2"
    result = dry_run.dry_run("SELECT 1", conninfo=url, max_cost=MAX_COST, max_plan_rows=MAX_ROWS)
    assert (result.outcome, result.rule) == (Outcome.FAILED, "db_unavailable")
