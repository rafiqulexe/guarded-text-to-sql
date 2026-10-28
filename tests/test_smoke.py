from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.default_limit <= settings.max_limit
    assert 0 <= settings.confidence_threshold <= settings.confidence_high <= 1


def test_app_starts() -> None:
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
