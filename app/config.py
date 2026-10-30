"""Settings loaded from the environment / `.env`.

Split so each part only needs what it uses:
- DatabaseSettings: DB URLs (migrations, DB scripts).
- GuardSettings: DB URLs + guard limits. No API key: guards never need one.
- LLMSettings: provider, key, models, rate-limit handling.
- Settings: everything, for the app and the pipeline runner.
"""

from datetime import date
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "gemini"]

# (GEN_MODEL, JUDGE_MODEL, SELF_CONSISTENCY_N, LLM_FALLBACK_MODEL) when not set in the environment.
PROVIDER_DEFAULTS: dict[str, tuple[str, str, int, str]] = {
    # Free tier: Flash for generation, lighter Flash-Lite for the judge (separate per-model
    # rate-limit bucket); N=2 keeps one question at ~3-5 calls.
    # Free-tier Flash models get overloaded (503/504): fall back to Flash-Lite.
    "gemini": ("gemini-3.8-flash", "gemini-3.5-flash-lite", 2, "gemini-3.5-flash-lite"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5", 3, ""),
}
MODEL_PREFIX: dict[str, str] = {"gemini": "gemini-", "anthropic": "claude-"}

# List prices, USD per 1M tokens (input, output). Used for `cost_usd_list` (what a run would cost
# at paid prices) and for `cost_usd` when LLM_BILLING=paid. Checked 2026-10-29 on
# ai.google.dev/gemini-api/docs/pricing (standard, paid tier; Gemini output includes thinking
# tokens) and the Anthropic model table. Note: gemini-3.8-flash / 3.6-flash double to
# 1.50 / 7.50 from 2027-01-01.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


class DatabaseSettings(_EnvSettings):
    """DB URLs only, so migrations and DB scripts run without an API key."""

    target_db_url: str = "postgresql://sqlguard_reader:reader_pw@localhost:5432/pagila"
    admin_db_url: str = "postgresql://postgres:postgres@localhost:5432/pagila"
    app_db_url: str = "postgresql://sqlguard_app:app_pw@localhost:5432/sqlguard_app"


class GuardSettings(DatabaseSettings):
    """Limits used by REWRITING / DRY_RUN / EXECUTING. Never needs an API key."""

    default_limit: int = Field(100, gt=0)
    max_limit: int = Field(1000, gt=0)
    statement_timeout_ms: int = Field(5000, gt=0)
    max_cost: float = Field(100000, gt=0)
    max_plan_rows: int = Field(1000000, gt=0)

    @model_validator(mode="after")
    def _limits_are_ordered(self) -> Self:
        if self.default_limit > self.max_limit:
            raise ValueError("DEFAULT_LIMIT must be <= MAX_LIMIT")
        return self


class LLMSettings(_EnvSettings):
    llm_provider: Provider = "gemini"
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gen_model: str = ""  # empty -> PROVIDER_DEFAULTS
    judge_model: str = ""
    llm_rpm: int = Field(10, gt=0)  # client-side cap: requests per minute, per model
    llm_max_retries: int = Field(5, ge=0)  # retries for 429 / overloaded
    llm_max_backoff_s: float = Field(60, gt=0)  # longer server retry-after = quota exhausted
    llm_timeout_s: float = Field(60, gt=0)  # per request; a hung call is a transient error
    llm_fallback_model: str | None = None  # None -> provider default; "none" disables
    llm_billing: Literal["free", "paid"] = "free"  # free tier: real cost_usd is 0

    @model_validator(mode="after")
    def _provider_is_configured(self) -> Self:
        gen_default, judge_default, _, fallback_default = PROVIDER_DEFAULTS[self.llm_provider]
        self.gen_model = self.gen_model or gen_default
        self.judge_model = self.judge_model or judge_default
        if self.llm_fallback_model is None:
            self.llm_fallback_model = fallback_default
        elif self.llm_fallback_model.lower() == "none":
            self.llm_fallback_model = ""
        prefix = MODEL_PREFIX[self.llm_provider]
        models = [("GEN_MODEL", self.gen_model), ("JUDGE_MODEL", self.judge_model)]
        if self.llm_fallback_model:
            models.append(("LLM_FALLBACK_MODEL", self.llm_fallback_model))
        for name, model in models:
            if not model.startswith(prefix):
                raise ValueError(
                    f"{name}={model!r} is not a {self.llm_provider} model (expected {prefix}*)"
                )
        if self.api_key is None:
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider} needs {self.llm_provider.upper()}_API_KEY"
            )
        return self

    @property
    def api_key(self) -> SecretStr | None:
        return self.gemini_api_key if self.llm_provider == "gemini" else self.anthropic_api_key


class Settings(GuardSettings, LLMSettings):
    max_repairs: int = Field(2, ge=0)
    self_consistency_n: int = Field(0, ge=0)  # 0 -> PROVIDER_DEFAULTS
    confidence_threshold: float = Field(0.7, ge=0, le=1)
    confidence_high: float = Field(0.85, ge=0, le=1)
    # Confidence scoring weights; must be >= 0 and sum to 1.
    # The judge dominates: a 'partial' verdict with every other signal perfect scores 0.675,
    # below the default confirmation threshold (0.7), so a doubtful answer is never auto-shown.
    score_weight_judge: float = 0.65
    score_weight_consistency: float = 0.15
    score_weight_repairs: float = 0.10
    score_weight_sanity: float = 0.10
    max_question_chars: int = Field(1000, gt=0)
    reference_date: date | None = None
    # First and last rental / payment date in the loaded data; the judge is given this range.
    data_start_date: date | None = None
    data_end_date: date | None = None

    @model_validator(mode="after")
    def _score_weights_are_valid(self) -> Self:
        weights = (
            self.score_weight_judge,
            self.score_weight_consistency,
            self.score_weight_repairs,
            self.score_weight_sanity,
        )
        if any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
            raise ValueError(f"SCORE_WEIGHT_* must be >= 0 and sum to 1, got {weights}")
        return self

    @model_validator(mode="after")
    def _data_dates_are_ordered(self) -> Self:
        start, end, ref = self.data_start_date, self.data_end_date, self.reference_date
        if start and end and start > end:
            raise ValueError(f"DATA_START_DATE {start} is after DATA_END_DATE {end}")
        if end and ref and end >= ref:
            raise ValueError(f"REFERENCE_DATE {ref} must be after DATA_END_DATE {end}")
        return self

    @model_validator(mode="after")
    def _self_consistency_default(self) -> Self:
        if self.self_consistency_n == 0:
            self.self_consistency_n = PROVIDER_DEFAULTS[self.llm_provider][2]
        return self


@lru_cache
def get_guard_settings() -> GuardSettings:
    return GuardSettings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
