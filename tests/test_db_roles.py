"""DB-level backstop: what `sqlguard_reader` can and cannot do (SCHEMA.md §2).

Needs the docker Postgres (`docker compose up -d`); skipped if it is not reachable.
"""

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from psycopg import errors
from psycopg.conninfo import make_conninfo

from app.config import DatabaseSettings

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST: dict[str, Any] = yaml.safe_load(
    (ROOT / "db" / "allowlist.yaml").read_text(encoding="utf-8")
)
READER_URL = DatabaseSettings().target_db_url


def _connect(url: str = READER_URL, **kwargs: Any) -> psycopg.Connection[Any]:
    return psycopg.connect(url, connect_timeout=3, **kwargs)


@pytest.fixture(scope="module")
def reachable() -> None:
    try:
        _connect().close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"target DB not reachable (docker compose up -d): {exc}")


@pytest.fixture
def reader(reachable: None) -> Iterator[psycopg.Connection[Any]]:
    with _connect(autocommit=True) as conn:
        yield conn


def _allowed() -> list[tuple[str, list[str]]]:
    return [(t, list(spec["columns"])) for t, spec in ALLOWLIST["tables"].items()]


def _restricted() -> list[tuple[str, str]]:
    return [(t, c) for t, spec in ALLOWLIST["tables"].items() for c in spec.get("restricted", [])]


pytestmark = pytest.mark.db


# --- the PLAN.md 1.3 checks ---------------------------------------------------

DENIED = [
    ("SELECT email FROM customer", errors.InsufficientPrivilege),
    ("DELETE FROM film", errors.ReadOnlySqlTransaction),
    ("SELECT pg_sleep(10)", errors.InsufficientPrivilege),
    ("SELECT pg_sleep_for('10s')", errors.InsufficientPrivilege),
    ("SELECT film_in_stock(1, 1)", errors.InsufficientPrivilege),
    # further backstops
    ("SELECT pg_sleep_until(now())", errors.InsufficientPrivilege),
    ("SELECT pg_read_file('/etc/passwd')", errors.InsufficientPrivilege),
    ("SELECT * FROM customer", errors.InsufficientPrivilege),
    ("SELECT count(*) FROM payment_p2022_01", errors.InsufficientPrivilege),
    ("SELECT * FROM customer_list", errors.InsufficientPrivilege),
    ("UPDATE film SET title = 'x'", errors.ReadOnlySqlTransaction),
    ("CREATE TABLE t (x int)", errors.ReadOnlySqlTransaction),
]


@pytest.mark.parametrize(("sql", "error"), DENIED, ids=[sql for sql, _ in DENIED])
def test_reader_denied(
    reader: psycopg.Connection[Any], sql: str, error: type[psycopg.Error]
) -> None:
    with pytest.raises(error):
        reader.execute(sql)


# --- grants match the allowlist -----------------------------------------------


@pytest.mark.parametrize(("table", "columns"), _allowed(), ids=[t for t, _ in _allowed()])
def test_allowlisted_columns_readable(
    reader: psycopg.Connection[Any], table: str, columns: list[str]
) -> None:
    reader.execute(f"SELECT {', '.join(columns)} FROM {table} LIMIT 1").fetchall()


@pytest.mark.parametrize(
    ("table", "column"), _restricted(), ids=[f"{t}.{c}" for t, c in _restricted()]
)
def test_restricted_columns_denied(
    reader: psycopg.Connection[Any], table: str, column: str
) -> None:
    with pytest.raises(errors.InsufficientPrivilege):
        reader.execute(f"SELECT {column} FROM {table} LIMIT 1")


def test_write_denied_even_in_read_write_transaction(reachable: None) -> None:
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ WRITE")
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute("DELETE FROM film")


def test_role_settings(reader: psycopg.Connection[Any]) -> None:
    row = reader.execute(
        "SELECT current_user, current_setting('default_transaction_read_only'),"
        " current_setting('statement_timeout'), current_setting('search_path')"
    ).fetchone()
    assert row == ("sqlguard_reader", "on", "5s", "public")


@pytest.mark.parametrize("dbname", ["postgres", "sqlguard_app"])
def test_reader_cannot_connect_to_other_dbs(reachable: None, dbname: str) -> None:
    url = make_conninfo(READER_URL, dbname=dbname)
    with pytest.raises(psycopg.OperationalError, match="CONNECT privilege"):
        _connect(url).close()


# --- generated grants are in sync (no DB needed) ------------------------------


def test_roles_sql_matches_allowlist() -> None:
    spec = importlib.util.spec_from_file_location("gen_grants", ROOT / "db" / "gen_grants.py")
    assert spec and spec.loader
    gen_grants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_grants)
    expected = gen_grants.render(ALLOWLIST)
    actual = (ROOT / "db" / "init" / "03_roles.sql").read_text(encoding="utf-8")
    assert actual == expected, "run: uv run python db/gen_grants.py"
