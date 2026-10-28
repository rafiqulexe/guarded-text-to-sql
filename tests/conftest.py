import os

import psycopg
import pytest

# Settings require an API key; tests must never need a real one.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app.config import DatabaseSettings


@pytest.fixture(scope="session")
def reader_url() -> str:
    """TARGET_DB_URL (the sqlguard_reader role); skips the test if Postgres is not running."""
    url = DatabaseSettings().target_db_url
    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"target DB not reachable (docker compose up -d): {exc}")
    return url
