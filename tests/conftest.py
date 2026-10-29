import os

import psycopg
import pytest

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


@pytest.fixture(scope="session")
def reader_url() -> str:
    """TARGET_DB_URL (the sqlguard_reader role); skips the test if Postgres is not running."""
    url = DatabaseSettings().target_db_url
    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"target DB not reachable (docker compose up -d): {exc}")
    return url
