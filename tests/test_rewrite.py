"""REWRITING: emit the *-expanded SQL, inject LIMIT if absent, clamp it if > MAX_LIMIT.
Only the outermost query is touched; ORDER BY and subquery LIMITs are kept."""

import pytest
import sqlglot
from sqlglot import exp

from app.pipeline import ast_guard, rewrite, schema_guard
from app.pipeline.result import Outcome, Stage

DEFAULT, MAX = 100, 1000


def _rewrite(sql: str) -> rewrite.RewriteResult:
    ast = ast_guard.check(sql)
    assert ast.passed and ast.expression is not None, ast.detail
    schema = schema_guard.check(ast.expression)
    assert schema.passed and schema.expression is not None, schema.detail
    result = rewrite.rewrite(schema.expression, default_limit=DEFAULT, max_limit=MAX)
    assert result.stage is Stage.REWRITING
    assert result.outcome is Outcome.PASSED, result.detail
    return result


def _outer_limit(sql: str) -> int | None:
    tree = sqlglot.parse_one(sql, read="postgres")
    limit = tree.args.get("limit")
    return int(limit.expression.this) if limit is not None else None


# --- (limit as written) -> (limit applied, injected, clamped) ----------------------

CASES = [
    ("SELECT title FROM film", 100, True, False),
    ("SELECT title FROM film LIMIT 5", 5, False, False),
    ("SELECT title FROM film LIMIT 0", 0, False, False),
    ("SELECT title FROM film LIMIT 1000", 1000, False, False),
    ("SELECT title FROM film LIMIT 1001", 1000, False, True),
    ("SELECT title FROM film LIMIT 50000", 1000, False, True),
    ("SELECT title FROM film LIMIT ALL", 1000, False, True),
    ("SELECT title FROM film LIMIT NULL", 1000, False, True),
    ("SELECT title FROM film LIMIT (SELECT 10)", 1000, False, True),
    ("SELECT title FROM film LIMIT 2 + 3", 1000, False, True),
    ("SELECT title FROM film FETCH FIRST 5 ROWS ONLY", 5, False, False),
    ("SELECT title FROM film FETCH FIRST ROW ONLY", 1, False, False),
    ("SELECT title FROM film FETCH FIRST 5000 ROWS ONLY", 1000, False, True),
]


@pytest.mark.parametrize(("sql", "limit", "injected", "clamped"), CASES, ids=[c[0] for c in CASES])
def test_limit(sql: str, limit: int, injected: bool, clamped: bool) -> None:
    result = _rewrite(sql)
    assert (result.limit, result.limit_injected, result.limit_clamped) == (limit, injected, clamped)
    assert result.limit_applied is (injected or clamped)
    assert _outer_limit(result.sql) == limit


def test_order_by_is_kept_before_the_injected_limit() -> None:
    result = _rewrite("SELECT title FROM film ORDER BY title DESC")
    assert result.sql.endswith("ORDER BY title DESC LIMIT 100")


def test_offset_is_kept() -> None:
    assert _rewrite("SELECT title FROM film LIMIT 10 OFFSET 20").sql.endswith("LIMIT 10 OFFSET 20")
    result = _rewrite("SELECT title FROM film OFFSET 5")
    assert (result.limit, result.limit_injected) == (100, True)
    assert "OFFSET 5" in result.sql


# --- only the outermost query is limited ---------------------------------------------


@pytest.mark.parametrize(
    ("sql", "inner"),
    [
        ("SELECT title FROM (SELECT title FROM film LIMIT 50000) AS s", "LIMIT 50000"),
        ("SELECT title FROM (SELECT title FROM film LIMIT 5) AS s", "LIMIT 5"),
        ("WITH t AS (SELECT title FROM film LIMIT 3000) SELECT title FROM t", "LIMIT 3000"),
        (
            "SELECT title FROM film WHERE film_id IN (SELECT film_id FROM inventory LIMIT 7)",
            "LIMIT 7",
        ),
    ],
)
def test_subquery_limits_untouched(sql: str, inner: str) -> None:
    result = _rewrite(sql)
    assert inner in result.sql
    assert (result.limit, result.limit_injected) == (100, True)
    assert _outer_limit(result.sql) == 100


def test_set_operation_is_limited_as_a_whole() -> None:
    result = _rewrite("SELECT title FROM film UNION SELECT first_name FROM actor")
    tree = sqlglot.parse_one(result.sql, read="postgres")
    assert isinstance(tree, exp.Union)
    assert int(tree.args["limit"].expression.this) == 100
    assert tree.left.args.get("limit") is None and tree.right.args.get("limit") is None

    clamped = _rewrite("SELECT title FROM film UNION SELECT first_name FROM actor LIMIT 9999")
    assert (clamped.limit, clamped.limit_clamped) == (1000, True)


def test_with_ties_is_bounded_by_an_outer_limit() -> None:
    """FETCH ... WITH TIES can return more than n rows, so it is wrapped and clamped."""
    result = _rewrite("SELECT title, length FROM film ORDER BY length FETCH FIRST 5 ROWS WITH TIES")
    assert (result.limit, result.limit_clamped) == (1000, True)
    assert _outer_limit(result.sql) == 1000
    assert "WITH TIES" in result.sql


# --- the emitted SQL is the checked, expanded tree -------------------------------------


def test_star_is_emitted_expanded_without_restricted_columns() -> None:
    result = _rewrite("SELECT * FROM customer")
    assert "*" not in result.sql
    assert "email" not in result.sql.lower()
    assert result.sql.endswith("LIMIT 100")


def test_comments_are_stripped() -> None:
    result = _rewrite("SELECT title /* note */ FROM film -- trailing")
    assert "/*" not in result.sql and "--" not in result.sql


@pytest.mark.parametrize("sql", [c[0] for c in CASES] + ["SELECT * FROM staff ORDER BY 1"])
def test_output_passes_ast_check_again(sql: str) -> None:
    result = _rewrite(sql)
    recheck = ast_guard.check(result.sql)
    assert recheck.passed, recheck.detail


def test_input_tree_is_not_mutated() -> None:
    ast = ast_guard.check("SELECT title FROM film")
    assert ast.expression is not None
    before = ast.expression.sql()
    rewrite.rewrite(ast.expression, default_limit=DEFAULT, max_limit=MAX)
    assert ast.expression.sql() == before


def test_default_limit_above_max_is_rejected() -> None:
    ast = ast_guard.check("SELECT title FROM film")
    assert ast.expression is not None
    with pytest.raises(ValueError):
        rewrite.rewrite(ast.expression, default_limit=2000, max_limit=MAX)


def test_limits_default_to_settings() -> None:
    ast = ast_guard.check("SELECT title FROM film")
    assert ast.expression is not None
    result = rewrite.rewrite(ast.expression)
    assert result.limit == 100  # DEFAULT_LIMIT in .env.example / config default
