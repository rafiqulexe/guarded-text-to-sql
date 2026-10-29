import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

# Unit tests never use a real key or the developer's model choices: environment variables
# win over .env, so these pin the LLM settings. Real-API checks live in scripts/live_llm_check.py.
os.environ.update(
    {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "GEN_MODEL": "gemini-3.8-flash",
        "JUDGE_MODEL": "gemini-3.5-flash-lite",
        "SELF_CONSISTENCY_N": "2",
    }
)

from app.config import DatabaseSettings
from app.db.models import QueryRun
from app.db.session import get_sessionmaker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def reader_url() -> str:
    """TARGET_DB_URL (the sqlguard_reader role); skips the test if Postgres is not running."""
    url = DatabaseSettings().target_db_url
    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"target DB not reachable (docker compose up -d): {exc}")
    return url


@pytest.fixture(scope="session")
def app_sessions() -> Iterator[sessionmaker[Session]]:
    """App DB (sqlguard_app) at the latest migration; skips if Postgres is not running."""
    try:
        psycopg.connect(DatabaseSettings().app_db_url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"app DB not reachable (docker compose up -d): {exc}")
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    yield get_sessionmaker()


@pytest.fixture
def clean_runs(app_sessions: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    """Deletes the query_runs (and, by cascade, guard_events) a test created."""
    started = datetime.now(UTC)
    yield app_sessions
    with app_sessions.begin() as s:
        s.execute(delete(QueryRun).where(QueryRun.created_at >= started))
