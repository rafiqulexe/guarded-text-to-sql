"""SCHEMA_CHECK (layer 2): every table/column must be allowlisted. Restricted (PII) columns are
BLOCKED, never repaired; unknown references go back for repair."""

import pytest
from sqlglot import exp

from app.allowlist import load_allowlist
from app.pipeline import ast_guard, schema_guard
from app.pipeline.result import GuardResult, Outcome, Stage

ALLOWLIST = load_allowlist()


def _check(sql: str) -> GuardResult:
    ast = ast_guard.check(sql)
    assert ast.passed and ast.expression is not None, f"AST_CHECK rejected test input: {ast}"
    result = schema_guard.check(ast.expression)
    assert result.stage is Stage.SCHEMA_CHECK
    return result


def _projection_names(result: GuardResult) -> list[str]:
    assert isinstance(result.expression, exp.Query)
    return [p.alias_or_name for p in result.expression.selects]


# --- allowed -----------------------------------------------------------------

ALLOWED = [
    "SELECT title, rating FROM film",
    "SELECT c.first_name, c.last_name FROM customer c",
    "SELECT c.first_name FROM customer AS c JOIN address AS a ON a.address_id = c.address_id"
    " WHERE a.district = 'Alberta'",
    "SELECT title, name FROM film JOIN language USING (language_id)",
    "WITH t AS (SELECT customer_id, first_name FROM customer) SELECT first_name FROM t",
    "WITH t AS (SELECT first_name AS n FROM customer) SELECT n FROM t ORDER BY n",
    "SELECT n FROM (SELECT first_name AS n FROM actor) AS sub",
    "SELECT title FROM film WHERE film_id IN (SELECT film_id FROM inventory)",
    "SELECT first_name FROM actor UNION SELECT first_name FROM customer",
    'SELECT "title" FROM "film"',
    "SELECT TITLE FROM FILM",
    "SELECT first_name AS email FROM customer ORDER BY email",
    "SELECT count(*) FROM customer",
    "SELECT sum(amount) FROM payment WHERE payment_date >= DATE '2026-07-01'",
    "SELECT c.name, count(*) AS n FROM category c JOIN film_category fc"
    " ON fc.category_id = c.category_id GROUP BY c.name ORDER BY n DESC",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql: str) -> None:
    result = _check(sql)
    assert result.outcome is Outcome.PASSED, result.detail
    assert result.expression is not None


# --- * expansion: allowlisted columns only -------------------


def test_select_star_from_customer_expands_without_email() -> None:
    result = _check("SELECT * FROM customer")
    assert (result.outcome, result.rule) == (Outcome.PASSED, "star_expanded")
    assert _projection_names(result) == list(ALLOWLIST.tables["customer"].columns)
    assert "email" not in _projection_names(result)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM staff",
        "SELECT s.* FROM staff s",
        "SELECT * FROM (SELECT * FROM staff) AS sub",
        "WITH t AS (SELECT * FROM staff) SELECT * FROM t",
    ],
)
def test_star_never_expands_to_restricted_columns(sql: str) -> None:
    result = _check(sql)
    assert result.outcome is Outcome.PASSED, result.detail
    assert result.expression is not None
    rendered = result.expression.sql(dialect="postgres")
    for column in ALLOWLIST.tables["staff"].restricted:
        assert column not in rendered.lower()
    assert "*" not in rendered


def test_star_in_join_expands_each_table_to_allowlisted_columns() -> None:
    result = _check("SELECT * FROM customer c JOIN address a ON a.address_id = c.address_id")
    assert result.passed
    names = _projection_names(result)
    assert "email" not in names and "phone" not in names
    assert set(names) == set(ALLOWLIST.tables["customer"].columns) | set(
        ALLOWLIST.tables["address"].columns
    )


def test_input_tree_is_not_mutated() -> None:
    ast = ast_guard.check("SELECT * FROM customer")
    assert ast.expression is not None
    before = ast.expression.sql()
    schema_guard.check(ast.expression)
    assert ast.expression.sql() == before


# --- BLOCKED: restricted (PII) columns, in every position and disguise -----------

RESTRICTED = [
    "SELECT email FROM customer",
    "SELECT c.email FROM customer c",
    "SELECT EMAIL FROM CUSTOMER",
    'SELECT "email" FROM customer',
    "SELECT first_name FROM customer WHERE email LIKE '%@example.com'",
    "SELECT first_name FROM customer ORDER BY email",
    "SELECT first_name FROM customer GROUP BY first_name, email",
    "SELECT count(DISTINCT email) FROM customer",
    "SELECT lower(email) AS e FROM customer",
    "SELECT x FROM (SELECT email AS x FROM customer) AS s",
    "WITH t AS (SELECT * FROM customer) SELECT email FROM t",
    "WITH t AS (SELECT * FROM customer) SELECT t.email FROM t",
    "WITH t AS (SELECT email FROM customer) SELECT 1 FROM t",
    "WITH t AS (SELECT * FROM staff), u AS (SELECT * FROM t) SELECT password FROM u",
    "SELECT phone FROM address",
    "SELECT a.phone FROM customer c JOIN address a ON a.address_id = c.address_id",
    "SELECT password FROM staff",
    "SELECT username FROM staff",
    "SELECT picture FROM staff",
    "SELECT s.email FROM staff s JOIN store st ON st.store_id = s.store_id",
    "SELECT email FROM customer UNION SELECT title FROM film",
    "SELECT title FROM film UNION SELECT password FROM staff",
    "SELECT first_name FROM customer WHERE EXISTS (SELECT 1 FROM staff s WHERE s.password = 'x')",
    "SELECT title FROM film WHERE title = (SELECT email FROM customer LIMIT 1)",
    "SELECT email FROM customer JOIN staff USING (store_id)",
    "SELECT c.first_name FROM customer c JOIN staff s ON s.email = c.first_name",
]


@pytest.mark.parametrize("sql", RESTRICTED)
def test_restricted_column_blocked(sql: str) -> None:
    result = _check(sql)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "restricted_column"), result.detail


# --- BLOCKED: whole-row references would read every column, PII included ----------

WHOLE_ROW = [
    "SELECT c FROM customer c",
    "SELECT customer FROM customer",
    "SELECT (c).email FROM customer c",
    "SELECT count(c.*) FROM customer c",
    "SELECT f FROM film f",
]


@pytest.mark.parametrize("sql", WHOLE_ROW)
def test_whole_row_reference_blocked(sql: str) -> None:
    result = _check(sql)
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "whole_row_reference"), result.detail


# --- repair: unknown (hallucinated) references ------------------------------------

UNKNOWN = [
    ("SELECT * FROM movies", "unknown_table"),
    ("SELECT title FROM films", "unknown_table"),
    ("SELECT * FROM payment_p2022_01", "unknown_table"),
    ("SELECT * FROM customer_list", "unknown_table"),
    ("SELECT f.title FROM film f JOIN reviews r ON r.film_id = f.film_id", "unknown_table"),
    ('SELECT * FROM "Film"', "unknown_table"),
    ("SELECT film_name FROM film", "unknown_column"),
    ("SELECT f.director FROM film f", "unknown_column"),
    ("SELECT last_update FROM film", "unknown_column"),
    ("SELECT fulltext FROM film", "unknown_column"),
    ("SELECT activebool FROM customer", "unknown_column"),
    ("SELECT address2 FROM address", "unknown_column"),
    ('SELECT "Title" FROM film', "unknown_column"),
    ("WITH t AS (SELECT title FROM film) SELECT rating FROM t", "unknown_column"),
    ("SELECT title FROM film WHERE box_office > 100", "unknown_column"),
    ("SELECT store_id FROM customer, staff", "unknown_column"),
]


@pytest.mark.parametrize(("sql", "rule"), UNKNOWN, ids=[s for s, _ in UNKNOWN])
def test_unknown_reference_repaired(sql: str, rule: str) -> None:
    result = _check(sql)
    assert (result.outcome, result.rule) == (Outcome.REPAIR, rule), result.detail


def test_restricted_beats_unknown() -> None:
    """Safety first: a query with both a PII column and a hallucinated table is BLOCKED."""
    result = _check(
        "SELECT c.email FROM customer c JOIN reviews r ON r.customer_id = c.customer_id"
    )
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "restricted_column")
