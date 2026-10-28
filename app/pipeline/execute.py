"""EXECUTING (layer 4): run guarded SQL as sqlguard_reader in a READ ONLY transaction with a
local statement_timeout, then roll back. Also the DB error -> outcome mapping (STATE.md §5)
shared with DRY_RUN.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import errors

from app.pipeline.result import GuardResult, Stage, blocked, failed, repair

CONNECT_TIMEOUT_S = 5


@contextmanager
def reader_transaction(conninfo: str, timeout_ms: int) -> Iterator[psycopg.Cursor[Any]]:
    """BEGIN READ ONLY; SET LOCAL statement_timeout; ...; ROLLBACK (SCHEMA.md §2)."""
    with psycopg.connect(conninfo, connect_timeout=CONNECT_TIMEOUT_S) as conn:
        conn.read_only = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                row = cur.execute("SHOW transaction_read_only").fetchone()
                if row is None or row[0] != "on":
                    raise RuntimeError("transaction is not read-only; refusing to run SQL")
                yield cur
        finally:
            conn.rollback()


def run_guarded(cur: psycopg.Cursor[Any], sql: str) -> None:
    """prepare=True makes the server reject anything but a single statement."""
    cur.execute(sql, prepare=True)


def classify_db_error(stage: Stage, exc: Exception) -> GuardResult:
    detail = _message(exc)
    if isinstance(exc, errors.QueryCanceled):
        return failed(stage, "timeout", f"statement_timeout exceeded: {detail}")
    if isinstance(exc, errors.InsufficientPrivilege):
        return blocked(stage, "insufficient_privilege", detail, critical=True)
    if isinstance(exc, errors.ReadOnlySqlTransaction):
        return blocked(stage, "read_only_violation", detail, critical=True)
    if isinstance(exc, errors.SyntaxError) and "multiple commands" in detail:
        return blocked(stage, "multiple_statements", detail, critical=True)
    if isinstance(exc, psycopg.OperationalError) and getattr(exc, "sqlstate", None) is None:
        return failed(stage, "db_unavailable", detail)
    if isinstance(exc, psycopg.Error) and (exc.sqlstate or "")[:2] in {"42", "22"}:
        # 42: undefined column/table/function, syntax; 22: data errors (e.g. division by zero).
        return repair(stage, "db_error", detail)
    return failed(stage, "db_error_unrecoverable", detail)


def _message(exc: Exception) -> str:
    if isinstance(exc, psycopg.Error) and exc.diag.message_primary:
        return exc.diag.message_primary
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
