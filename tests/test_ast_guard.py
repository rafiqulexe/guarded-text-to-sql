"""AST_CHECK (layer 1): parse, single statement, read-only query shape, function allowlist,
system catalogs."""

import pytest
import sqlglot
from sqlglot import exp

from app.pipeline import ast_guard
from app.pipeline.result import Outcome, Stage


def _check(sql: str) -> tuple[Outcome, str]:
    result = ast_guard.check(sql)
    assert result.stage is Stage.AST_CHECK
    return result.outcome, result.rule


# --- allowed -----------------------------------------------------------------

ALLOWED = [
    "SELECT title FROM film",
    "select title from film;",
    "sElEcT TiTlE FrOm FiLm",
    "SELECT f.title, count(*) AS n FROM film f JOIN film_actor fa ON fa.film_id = f.film_id"
    " WHERE f.rating = 'PG' GROUP BY f.title HAVING count(*) > 2 ORDER BY n DESC LIMIT 5",
    "WITH top AS (SELECT film_id FROM inventory) SELECT count(*) FROM top",
    "SELECT title FROM film WHERE rating = 'G' UNION SELECT title FROM film WHERE rating = 'PG'",
    "SELECT film_id FROM film INTERSECT SELECT film_id FROM inventory",
    "(SELECT title FROM film)",
    "SELECT public.film.title FROM public.film",
    "SELECT title FROM film -- trailing comment",
    "SELECT /* inline comment */ title FROM film",
    "SELECT CASE WHEN length > 120 THEN 'long' ELSE 'short' END, length::text,"
    " CAST(rental_rate AS int) FROM film",
    "SELECT EXTRACT(year FROM payment_date), date_part('month', payment_date),"
    " date_trunc('month', payment_date), to_char(payment_date, 'YYYY-MM') FROM payment",
    "SELECT now(), current_date",
    "SELECT round(avg(amount), 2), sum(amount), min(amount), max(amount) FROM payment",
    "SELECT lower(first_name), upper(last_name), trim(first_name), length(last_name),"
    " concat(first_name, ' ', last_name), coalesce(first_name, ''), nullif(last_name, '')"
    " FROM actor",
    "SELECT title FROM film WHERE special_features @> ARRAY['Trailers']",
    "SELECT title FROM film WHERE 'Trailers' = ANY(special_features)",
    "SELECT title FROM film f WHERE EXISTS (SELECT 1 FROM inventory i WHERE i.film_id = f.film_id)",
    "SELECT count(*) FILTER (WHERE amount > 5) FROM payment",
    "SELECT rental_id FROM rental WHERE rental_date >= DATE '2026-07-29' - INTERVAL '30 days'",
    "SELECT title FROM film WHERE title ILIKE 'a%' AND length BETWEEN 60 AND 90",
    "SELECT title FROM (SELECT title FROM film) AS sub",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql: str) -> None:
    result = ast_guard.check(sql)
    assert result.outcome is Outcome.PASSED, result.detail
    assert result.expression is not None


# --- BLOCKED (never repaired) ---------------------------------------------------

BLOCKED = [
    # multiple statements / comment smuggling
    ("SELECT 1; DROP TABLE film", "single_statement"),
    ("SELECT 1; SELECT 2", "single_statement"),
    ("SELECT 1 /* harmless */; DELETE FROM film", "single_statement"),
    ("SELECT 1 -- comment\n; DROP TABLE film", "single_statement"),
    ("SELECT title FROM film;\nUPDATE film SET title = 'x'", "single_statement"),
    # non-SELECT roots
    ("DELETE FROM film", "select_only"),
    ("UPDATE film SET title = 'x'", "select_only"),
    ("INSERT INTO film (title) VALUES ('x')", "select_only"),
    ("TRUNCATE film", "select_only"),
    ("DROP TABLE film", "select_only"),
    ("CREATE TABLE t (x int)", "select_only"),
    ("ALTER TABLE film ADD COLUMN x int", "select_only"),
    ("GRANT SELECT ON customer TO public", "select_only"),
    ("COPY film TO STDOUT", "select_only"),
    ("COPY (SELECT 1) TO PROGRAM 'rm -rf /'", "select_only"),
    ("SET statement_timeout = 0", "select_only"),
    ("SET ROLE postgres", "select_only"),
    ("RESET ALL", "select_only"),
    ("BEGIN", "select_only"),
    ("COMMIT", "select_only"),
    ("CALL do_something()", "select_only"),
    ("DO $$ BEGIN PERFORM pg_sleep(10); END $$", "select_only"),
    ("EXPLAIN ANALYZE DELETE FROM film", "select_only"),
    ("VACUUM film", "select_only"),
    ("VALUES (1)", "select_only"),
    ("TABLE film", "select_only"),
    # data-modifying CTEs
    ("WITH x AS (DELETE FROM film RETURNING *) SELECT * FROM x", "data_modifying_cte"),
    (
        "WITH x AS (UPDATE film SET title = 'x' RETURNING film_id) SELECT * FROM x",
        "data_modifying_cte",
    ),
    (
        "WITH x AS (INSERT INTO actor (first_name) VALUES ('a') RETURNING actor_id)"
        " SELECT * FROM x",
        "data_modifying_cte",
    ),
    (
        "WITH a AS (SELECT 1), b AS (DELETE FROM film RETURNING film_id) SELECT * FROM a",
        "data_modifying_cte",
    ),
    # recursive CTEs
    (
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) SELECT n FROM t",
        "recursive_cte",
    ),
    ("with recursive t AS (SELECT 1) SELECT * FROM t", "recursive_cte"),
    # SELECT ... INTO / row locks
    ("SELECT * INTO film_copy FROM film", "select_into"),
    ("SELECT title FROM film FOR UPDATE", "locking_clause"),
    ("SELECT title FROM film FOR SHARE", "locking_clause"),
    ("SELECT title FROM film FOR NO KEY UPDATE", "locking_clause"),
    # functions outside the allowlist, in every position and spelling
    ("SELECT pg_sleep(10)", "function_not_allowed"),
    ("SELECT PG_SLEEP(10)", "function_not_allowed"),
    ('SELECT "pg_sleep"(10)', "function_not_allowed"),
    ("SELECT pg_catalog.pg_sleep(10)", "function_not_allowed"),
    ("SELECT pg_sleep_for('10s')", "function_not_allowed"),
    ("SELECT pg_sleep_until(now() + interval '10 seconds')", "function_not_allowed"),
    ("SELECT film_in_stock(1, 1)", "function_not_allowed"),
    ("SELECT public.film_in_stock(1, 1)", "function_not_allowed"),
    ("SELECT * FROM film_in_stock(1, 1)", "function_not_allowed"),
    ("SELECT pg_read_file('/etc/passwd')", "function_not_allowed"),
    ("SELECT dblink('host=evil', 'SELECT 1')", "function_not_allowed"),
    ("SELECT lo_import('/etc/passwd')", "function_not_allowed"),
    ("SELECT current_setting('data_directory')", "function_not_allowed"),
    ("SELECT set_config('statement_timeout', '0', false)", "function_not_allowed"),
    ("SELECT * FROM generate_series(1, 1000000000)", "function_not_allowed"),
    ("SELECT string_agg(title, ',') FROM film", "function_not_allowed"),
    ("SELECT title FROM film WHERE pg_sleep(10) IS NOT NULL", "function_not_allowed"),
    ("SELECT title FROM film ORDER BY pg_sleep(1)", "function_not_allowed"),
    ("SELECT count(pg_sleep(1)) FROM film", "function_not_allowed"),
    ("SELECT f.title FROM film f JOIN inventory i ON pg_sleep(1) IS NULL", "function_not_allowed"),
    ("SELECT * FROM (SELECT pg_sleep(10)) s", "function_not_allowed"),
    ("WITH s AS (SELECT pg_sleep(10)) SELECT * FROM s", "function_not_allowed"),
    ("SELECT title FROM film UNION SELECT pg_read_file('/etc/passwd')", "function_not_allowed"),
    # system catalogs and other schemas
    ("SELECT * FROM information_schema.tables", "system_catalog"),
    ("SELECT * FROM INFORMATION_SCHEMA.TABLES", "system_catalog"),
    ("SELECT usename, passwd FROM pg_catalog.pg_shadow", "system_catalog"),
    ('SELECT * FROM "pg_catalog"."pg_shadow"', "system_catalog"),
    ("SELECT * FROM pg_shadow", "system_catalog"),
    ("SELECT * FROM PG_USER", "system_catalog"),
    ("SELECT query FROM pg_stat_activity", "system_catalog"),
    ("SELECT * FROM pg_toast.pg_toast_1", "system_catalog"),
    ("SELECT * FROM secret_schema.film", "system_catalog"),
    ('SELECT * FROM "Public".film', "system_catalog"),
    ("SELECT * FROM pagila.public.film", "system_catalog"),
    ("SELECT title FROM film WHERE film_id IN (SELECT oid FROM pg_class)", "system_catalog"),
    ("WITH t AS (SELECT * FROM information_schema.columns) SELECT * FROM t", "system_catalog"),
    ("SELECT f.title FROM film f JOIN pg_catalog.pg_roles r ON true", "system_catalog"),
]


@pytest.mark.parametrize(("sql", "rule"), BLOCKED, ids=[s for s, _ in BLOCKED])
def test_blocked(sql: str, rule: str) -> None:
    assert _check(sql) == (Outcome.BLOCKED, rule)


# --- repair (back to GENERATING) --------------------------------

REPAIR = [
    ("SELEC title FROM film", "parse_error"),
    ("SELECT (title FROM film", "parse_error"),
    ("SELECT title FROM film WHERE", "parse_error"),
    ("", "empty_sql"),
    ("   ;  ", "empty_sql"),
    ("-- just a comment", "empty_sql"),
]


@pytest.mark.parametrize(("sql", "rule"), REPAIR, ids=[repr(s) for s, _ in REPAIR])
def test_repair(sql: str, rule: str) -> None:
    assert _check(sql) == (Outcome.REPAIR, rule)


# --- the checked tree is what runs: lexer tricks can't smuggle a second statement ----


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 /* /* nested */ ; DELETE FROM film; */",
        "SELECT $$; DROP TABLE film; $$",
        "SELECT E'\\'; DROP TABLE film; --'",
        "SELECT 'it''s; DROP TABLE film'",
    ],
)
def test_lexer_tricks_never_yield_a_second_statement(sql: str) -> None:
    result = ast_guard.check(sql)
    if result.outcome is Outcome.PASSED:
        assert result.expression is not None
        # Only SQL regenerated from the checked tree is ever executed; it must re-parse
        # as exactly one SELECT, with the dangerous text left inside a string literal.
        regenerated = result.expression.sql(dialect="postgres", comments=False)
        statements = [s for s in sqlglot.parse(regenerated, read="postgres") if s is not None]
        assert len(statements) == 1
        assert isinstance(statements[0], exp.Select)
        assert not list(statements[0].find_all(exp.Drop, exp.Delete))
    else:
        assert result.outcome in (Outcome.BLOCKED, Outcome.REPAIR)


def test_function_allowlist_comes_from_allowlist_yaml() -> None:
    from app.allowlist import load_allowlist, parse_allowlist

    base = load_allowlist()
    without_round = parse_allowlist(
        {
            "tables": {n: {"columns": t.columns} for n, t in base.tables.items()},
            "functions": sorted(base.functions - {"round"}),
        }
    )
    assert ast_guard.check("SELECT round(1.5)").passed
    assert ast_guard.check("SELECT round(1.5)", without_round).rule == "function_not_allowed"


def test_function_name_is_checked_as_executed() -> None:
    """The tree is re-rendered before execution; check the name that will actually run."""
    from app.allowlist import load_allowlist, parse_allowlist

    # A call shape sqlglot maps to an internal node that renders as a non-allowlisted name.
    result = ast_guard.check("SELECT cast(title, 'YYYY') FROM film")
    assert (result.outcome, result.rule) == (Outcome.BLOCKED, "function_not_allowed")

    # now() executes as CURRENT_TIMESTAMP, so it is allowed only via that allowlist entry.
    base = load_allowlist()
    no_current_timestamp = parse_allowlist(
        {
            "tables": {n: {"columns": t.columns} for n, t in base.tables.items()},
            "functions": sorted(base.functions - {"current_timestamp"}),
        }
    )
    assert ast_guard.check("SELECT now()").passed
    assert not ast_guard.check("SELECT now()", no_current_timestamp).passed


# --- syntax-exempt constant: small, explicit, and never a free pass for its contents ------


def test_syntax_exempt_list_is_exactly_this() -> None:
    """Pinned so that widening the exemption is always a deliberate, reviewed change."""
    assert set(ast_guard.SYNTAX_NODES) == {
        exp.Case,
        exp.If,
        exp.Array,
        exp.Exists,
        exp.And,
        exp.Or,
        exp.ArrayContainsAll,
        exp.ArrayContainedBy,
        exp.ArrayOverlaps,
    }
    assert len(ast_guard.SYNTAX_NODES) == len(set(ast_guard.SYNTAX_NODES))
    assert exp.Binary not in ast_guard.SYNTAX_NODES and exp.Func not in ast_guard.SYNTAX_NODES


SYNTAX_EXAMPLES = [
    ("CASE / WHEN", "SELECT CASE WHEN length > 120 THEN 'long' ELSE 'short' END FROM film"),
    ("ARRAY[...]", "SELECT ARRAY['a', 'b']"),
    ("ARRAY(subquery)", "SELECT ARRAY(SELECT title FROM film)"),
    ("EXISTS", "SELECT 1 WHERE EXISTS (SELECT 1 FROM film)"),
    ("AND / OR", "SELECT title FROM film WHERE (length > 60 AND rating = 'G') OR length < 50"),
    ("@>", "SELECT title FROM film WHERE special_features @> ARRAY['Trailers']"),
    ("<@", "SELECT title FROM film WHERE special_features <@ ARRAY['Trailers']"),
    ("&&", "SELECT title FROM film WHERE special_features && ARRAY['Trailers']"),
]


@pytest.mark.parametrize(
    "sql", [s for _, s in SYNTAX_EXAMPLES], ids=[n for n, _ in SYNTAX_EXAMPLES]
)
def test_each_exempt_construct_is_allowed(sql: str) -> None:
    result = ast_guard.check(sql)
    assert result.passed, result.detail


# sqlglot models these as "operators" too, but they are function calls (or render as one).
FUNCTION_LIKE_OPERATORS = [
    "SELECT corr(rental_rate, length) FROM film",
    "SELECT array_position(special_features, 'Trailers') FROM film",
    "SELECT 2 ^ 3",  # renders as POWER(2, 3)
    "SELECT if(true, 1, 2)",  # IF() outside CASE is a function call
    'SELECT title COLLATE "C" FROM film',
]


@pytest.mark.parametrize("sql", FUNCTION_LIKE_OPERATORS)
def test_function_like_operators_are_not_exempt(sql: str) -> None:
    assert _check(sql) == (Outcome.BLOCKED, "function_not_allowed")


HIDDEN_BANNED = [
    # inside CASE: condition, THEN, ELSE, nested CASE
    "SELECT CASE WHEN pg_sleep(10) IS NULL THEN 1 ELSE 0 END",
    "SELECT CASE WHEN length > 60 THEN pg_read_file('/etc/passwd') END FROM film",
    "SELECT CASE WHEN length > 60 THEN 'x' ELSE film_in_stock(1, 1)::text END FROM film",
    "SELECT CASE WHEN true THEN CASE WHEN false THEN 1 ELSE pg_sleep(1)::int END END",
    "SELECT title FROM film ORDER BY CASE WHEN pg_sleep(1) IS NULL THEN 1 END",
    # inside the other exempt constructs
    "SELECT ARRAY[pg_sleep(10)]",
    "SELECT ARRAY(SELECT pg_read_file('/etc/passwd'))",
    "SELECT 1 WHERE EXISTS (SELECT pg_sleep(10))",
    "SELECT title FROM film WHERE length > 60 AND pg_sleep(1) IS NOT NULL",
    "SELECT title FROM film WHERE length > 60 OR pg_sleep(1) IS NOT NULL",
    "SELECT title FROM film WHERE special_features @> ARRAY[current_setting('data_directory')]",
    "SELECT title FROM film WHERE special_features && ARRAY[pg_read_file('/etc/passwd')]",
    # inside a set-operation branch (first, second, nested, INTERSECT/EXCEPT)
    "SELECT pg_sleep(10) UNION SELECT 1",
    "SELECT title FROM film UNION ALL SELECT pg_read_file('/etc/passwd')",
    "SELECT title FROM film UNION SELECT title FROM film UNION SELECT dblink('x', 'y')::text",
    "SELECT film_id FROM film INTERSECT SELECT film_in_stock(1, 1)",
    "SELECT film_id FROM film EXCEPT SELECT CASE WHEN pg_sleep(1) IS NULL THEN 1 END",
    "(SELECT title FROM film) UNION (SELECT CASE WHEN true THEN pg_read_file('/x') END)",
]


@pytest.mark.parametrize("sql", HIDDEN_BANNED)
def test_banned_function_hidden_in_syntax_or_set_branch_is_blocked(sql: str) -> None:
    assert _check(sql) == (Outcome.BLOCKED, "function_not_allowed")
