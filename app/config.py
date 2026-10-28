"""Settings loaded from the environment / `.env`."""

from datetime import date
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    anthropic_api_key: SecretStr
    gen_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-haiku-5-5"

    target_db_url: str = "postgresql://sqlguard_reader:reader_pw@localhost:5432/pagila"
    admin_db_url: str = "postgresql://postgres:postgres@localhost:5432/pagila"
    app_db_url: str = "postgresql://sqlguard_app:app_pw@localhost:5432/sqlguard_app"

    default_limit: int = Field(100, gt=0)
    max_limit: int = Field(1000, gt=0)
    statement_timeout_ms: int = Field(5000, gt=0)
    max_cost: float = Field(100000, gt=0)
    max_plan_rows: int = Field(1000000, gt=0)
    max_repairs: int = Field(2, ge=0)
    self_consistency_n: int = Field(3, ge=1)
    confidence_threshold: float = Field(0.7, ge=0, le=1)
    confidence_high: float = Field(0.85, ge=0, le=1)
    max_question_chars: int = Field(1000, gt=0)
    reference_date: date | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
