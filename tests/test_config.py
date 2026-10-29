"""Settings split: guards need no API key; the LLM part validates provider, key and models."""

from typing import Any

import pytest
from pydantic import ValidationError

from app.config import GuardSettings, LLMSettings, Settings

KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GEN_MODEL", "JUDGE_MODEL", "SELF_CONSISTENCY_N")


@pytest.fixture
def no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*KEYS, "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)


def test_guard_settings_need_no_api_key(no_llm_env: None) -> None:
    settings = GuardSettings(_env_file=None)
    assert (settings.default_limit, settings.max_limit) == (100, 1000)
    assert not hasattr(settings, "gemini_api_key")


def test_guard_limits_must_be_ordered(no_llm_env: None) -> None:
    with pytest.raises(ValidationError, match="DEFAULT_LIMIT"):
        GuardSettings(_env_file=None, default_limit=2000, max_limit=1000)


def test_gemini_is_the_default_provider_with_free_tier_models(no_llm_env: None) -> None:
    settings = Settings(_env_file=None, gemini_api_key="k")
    assert settings.llm_provider == "gemini"
    assert (settings.gen_model, settings.judge_model) == (
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    )
    assert settings.self_consistency_n == 2


def test_anthropic_defaults(no_llm_env: None) -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    assert (settings.gen_model, settings.judge_model) == ("claude-sonnet-5-5", "claude-haiku-5-5")
    assert settings.self_consistency_n == 3


def test_explicit_n_wins(no_llm_env: None) -> None:
    settings = Settings(_env_file=None, gemini_api_key="k", self_consistency_n=3)
    assert settings.self_consistency_n == 3


@pytest.mark.parametrize(
    ("provider", "key_field"), [("gemini", "anthropic_api_key"), ("anthropic", "gemini_api_key")]
)
def test_key_for_the_selected_provider_is_required(
    no_llm_env: None, provider: str, key_field: str
) -> None:
    only_the_other_key: dict[str, Any] = {"llm_provider": provider, key_field: "k"}
    with pytest.raises(ValidationError, match=f"{provider.upper()}_API_KEY"):
        LLMSettings(_env_file=None, **only_the_other_key)


def test_model_must_match_provider(no_llm_env: None) -> None:
    with pytest.raises(ValidationError, match="not a gemini model"):
        LLMSettings(_env_file=None, gemini_api_key="k", gen_model="claude-sonnet-5-5")


def test_unknown_provider_rejected(no_llm_env: None) -> None:
    with pytest.raises(ValidationError):
        LLMSettings(_env_file=None, llm_provider="openai", gemini_api_key="k")


def test_fallback_model_defaults_by_provider(no_llm_env: None) -> None:
    gemini = LLMSettings(_env_file=None, gemini_api_key="k")
    claude = LLMSettings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    assert gemini.llm_fallback_model == "gemini-3.5-flash-lite"
    assert claude.llm_fallback_model == ""


def test_fallback_can_be_disabled_or_must_match_provider(no_llm_env: None) -> None:
    off = LLMSettings(_env_file=None, gemini_api_key="k", llm_fallback_model="none")
    assert off.llm_fallback_model == ""
    with pytest.raises(ValidationError, match="LLM_FALLBACK_MODEL"):
        LLMSettings(_env_file=None, gemini_api_key="k", llm_fallback_model="claude-haiku-5-5")
