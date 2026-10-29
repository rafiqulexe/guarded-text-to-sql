"""Provider-neutral LLM client, tested against fake SDK objects (no network, no key)."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from app.config import LLMSettings
from app.pipeline.llm import (
    AnthropicClient,
    ErrorKind,
    GeminiClient,
    LLMError,
    SlidingWindowLimiter,
    make_client,
)


class Answer(BaseModel):
    value: int


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _usage(prompt: int = 10, out: int = 5, thoughts: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thoughts
    )


class FakeGemini:
    """Stands in for genai.Client: `.aio.models.generate_content` replays scripted results."""

    def __init__(self, *results: Any) -> None:
        self.results: list[Any] = list(results)
        self.calls: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _ok(value: int = 7) -> SimpleNamespace:
    return SimpleNamespace(parsed=Answer(value=value), text=None, usage_metadata=_usage())


def _gemini_error(code: int, status: str, details: list[dict[str, Any]] | None = None) -> Exception:
    body = {"error": {"code": code, "status": status, "message": "x", "details": details or []}}
    return genai_errors.APIError(code, body)


def _retry_info(seconds: str) -> dict[str, Any]:
    return {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": seconds}


def _gemini(fake: FakeGemini, sleeps: Sleeps, **kwargs: Any) -> GeminiClient:
    options = {"rpm": 1000, "max_retries": 3, "max_backoff_s": 60} | kwargs
    return GeminiClient("k", sdk=fake, sleep=sleeps, **options)


async def _ask(client: Any) -> Any:
    return await client.structured(
        model="gemini-3.8-flash", system="sys", user="usr", schema=Answer, temperature=0.4
    )


# --- gemini: request shape, parsing, usage ---------------------------------------------------


async def test_gemini_structured_output_and_usage() -> None:
    fake = FakeGemini(_ok(42))
    response = await _ask(_gemini(fake, Sleeps()))
    assert response.parsed == Answer(value=42)
    assert (response.input_tokens, response.output_tokens, response.attempts) == (10, 8, 1)
    config = fake.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer
    assert (config.system_instruction, config.temperature) == ("sys", 0.4)
    assert (fake.calls[0]["model"], fake.calls[0]["contents"]) == ("gemini-3.8-flash", "usr")


async def test_gemini_falls_back_to_validating_the_json_text() -> None:
    fake = FakeGemini(SimpleNamespace(parsed=None, text='{"value": 3}', usage_metadata=_usage()))
    assert (await _ask(_gemini(fake, Sleeps()))).parsed == Answer(value=3)


async def test_invalid_output_is_retried_once_then_fails() -> None:
    bad = SimpleNamespace(parsed=None, text="not json", usage_metadata=_usage())
    fake, sleeps = FakeGemini(bad, bad, _ok()), Sleeps()
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, sleeps))
    assert raised.value.kind is ErrorKind.TRANSIENT
    assert len(fake.calls) == 2  # first try + 1 retry (STATE.md)


# --- gemini: 429 handling ---------------------------------------------------------------------


async def test_429_waits_the_server_retry_delay_then_succeeds() -> None:
    limited = _gemini_error(429, "RESOURCE_EXHAUSTED", [_retry_info("7s")])
    fake, sleeps = FakeGemini(limited, limited, _ok()), Sleeps()
    response = await _ask(_gemini(fake, sleeps))
    assert response.attempts == 3
    assert sleeps.calls == [7.0, 7.0]


async def test_429_without_delay_backs_off_exponentially() -> None:
    limited = _gemini_error(429, "RESOURCE_EXHAUSTED")
    fake, sleeps = FakeGemini(limited, limited, limited, _ok()), Sleeps()
    await _ask(_gemini(fake, sleeps))
    for got, base in zip(sleeps.calls, (2, 4, 8), strict=True):
        assert 0.8 * base <= got <= 1.2 * base


async def test_429_gives_up_after_max_retries() -> None:
    limited = _gemini_error(429, "RESOURCE_EXHAUSTED", [_retry_info("1s")])
    fake = FakeGemini(*[limited] * 10)
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, Sleeps(), max_retries=3))
    assert raised.value.kind is ErrorKind.RATE_LIMITED
    assert len(fake.calls) == 4


async def test_long_retry_delay_means_quota_exhausted_no_wait() -> None:
    fake, sleeps = (
        FakeGemini(_gemini_error(429, "RESOURCE_EXHAUSTED", [_retry_info("3600s")])),
        Sleeps(),
    )
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, sleeps))
    assert raised.value.kind is ErrorKind.QUOTA_EXHAUSTED
    assert (len(fake.calls), sleeps.calls) == (1, [])


async def test_daily_quota_violation_is_not_retried() -> None:
    violation = {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    }
    fake, sleeps = (
        FakeGemini(_gemini_error(429, "RESOURCE_EXHAUSTED", [violation, _retry_info("20s")])),
        Sleeps(),
    )
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, sleeps))
    assert raised.value.kind is ErrorKind.QUOTA_EXHAUSTED
    assert (len(fake.calls), sleeps.calls) == (1, [])


# --- gemini: other errors -----------------------------------------------------------------------


async def test_503_overloaded_is_retried_like_a_rate_limit() -> None:
    overloaded = _gemini_error(503, "UNAVAILABLE")
    fake = FakeGemini(overloaded, overloaded, _ok())
    assert (await _ask(_gemini(fake, Sleeps()))).attempts == 3


async def test_500_is_retried_once() -> None:
    fake = FakeGemini(*[_gemini_error(500, "INTERNAL")] * 5)
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, Sleeps()))
    assert raised.value.kind is ErrorKind.TRANSIENT
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("code", "status"), [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (404, "NOT_FOUND")]
)
async def test_client_errors_are_fatal_and_not_retried(code: int, status: str) -> None:
    fake = FakeGemini(_gemini_error(code, status), _ok())
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini(fake, Sleeps()))
    assert raised.value.kind is ErrorKind.FATAL
    assert len(fake.calls) == 1


# --- rate limiter ---------------------------------------------------------------------------------


async def test_limiter_lets_a_burst_through_then_waits_for_the_window() -> None:
    now = [0.0]
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = SlidingWindowLimiter(2, clock=lambda: now[0], sleep=sleep)
    await limiter.acquire()
    await limiter.acquire()  # N=2 parallel candidates: no wait
    assert sleeps == []
    now[0] = 15.0
    await limiter.acquire()  # third call in the same minute waits until t=60
    assert sleeps == [45.0]


async def test_each_model_has_its_own_window() -> None:
    fake, sleeps = FakeGemini(_ok(), _ok()), Sleeps()
    client = _gemini(fake, sleeps, rpm=1)
    await client.structured(
        model="gemini-3.8-flash", system="s", user="u", schema=Answer, temperature=0
    )
    await client.structured(
        model="gemini-3.5-flash-lite", system="s", user="u", schema=Answer, temperature=0
    )
    assert sleeps.calls == []


# --- anthropic ----------------------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self, *results: Any) -> None:
        self.results: list[Any] = list(results)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(parse=self._parse)

    async def _parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _message(value: int = 7, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=Answer(value=value) if stop_reason != "refusal" else None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=4),
    )


def _anthropic_error(
    cls: type[anthropic.APIStatusError], status: int, retry_after: str | None = None
) -> Exception:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx2.Response(status, headers=headers, request=httpx2.Request("POST", "https://x"))
    return cls("error", response=response, body=None)


def _claude(fake: FakeAnthropic, sleeps: Sleeps) -> AnthropicClient:
    return AnthropicClient("k", sdk=fake, sleep=sleeps, rpm=1000, max_retries=3, max_backoff_s=60)


async def test_anthropic_uses_parse_and_sends_no_temperature() -> None:
    fake = FakeAnthropic(_message(5))
    response = await _claude(fake, Sleeps()).structured(
        model="claude-sonnet-5-5", system="sys", user="usr", schema=Answer, temperature=0.4
    )
    assert response.parsed == Answer(value=5)
    assert (response.input_tokens, response.output_tokens) == (11, 4)
    call = fake.calls[0]
    assert call["output_format"] is Answer
    assert "temperature" not in call and "tool_choice" not in call
    assert call["messages"] == [{"role": "user", "content": "usr"}]


async def test_anthropic_rate_limit_honours_retry_after() -> None:
    limited = _anthropic_error(anthropic.RateLimitError, 429, retry_after="3")
    fake, sleeps = FakeAnthropic(limited, _message()), Sleeps()
    await _claude(fake, sleeps).structured(
        model="claude-sonnet-5-5", system="s", user="u", schema=Answer, temperature=0
    )
    assert sleeps.calls == [3.0]


async def test_anthropic_overloaded_529_is_retried() -> None:
    fake = FakeAnthropic(_anthropic_error(anthropic.APIStatusError, 529), _message())
    response = await _claude(fake, Sleeps()).structured(
        model="claude-sonnet-5-5", system="s", user="u", schema=Answer, temperature=0
    )
    assert response.attempts == 2


async def test_anthropic_refusal_is_fatal() -> None:
    fake = FakeAnthropic(_message(stop_reason="refusal"))
    with pytest.raises(LLMError) as raised:
        await _claude(fake, Sleeps()).structured(
            model="claude-sonnet-5-5", system="s", user="u", schema=Answer, temperature=0
        )
    assert raised.value.kind is ErrorKind.FATAL


async def test_anthropic_bad_request_is_fatal() -> None:
    fake = FakeAnthropic(_anthropic_error(anthropic.BadRequestError, 400), _message())
    with pytest.raises(LLMError) as raised:
        await _claude(fake, Sleeps()).structured(
            model="claude-sonnet-5-5", system="s", user="u", schema=Answer, temperature=0
        )
    assert raised.value.kind is ErrorKind.FATAL
    assert len(fake.calls) == 1


# --- factory ---------------------------------------------------------------------------------


def test_factory_picks_the_configured_provider() -> None:
    gemini = make_client(LLMSettings(_env_file=None, llm_provider="gemini", gemini_api_key="k"))
    claude = make_client(
        LLMSettings(
            _env_file=None,
            llm_provider="anthropic",
            anthropic_api_key="k",
            gen_model="claude-sonnet-5-5",
            judge_model="claude-haiku-5-5",
        )
    )
    assert (gemini.provider, claude.provider) == ("gemini", "anthropic")


async def test_count_tokens_on_both_providers() -> None:
    async def gemini_count(**kwargs: Any) -> Any:
        assert kwargs == {"model": "gemini-3.8-flash", "contents": "hello"}
        return SimpleNamespace(total_tokens=12)

    async def claude_count(**kwargs: Any) -> Any:
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        return SimpleNamespace(input_tokens=9)

    gemini_sdk = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(count_tokens=gemini_count))
    )
    claude_sdk = SimpleNamespace(messages=SimpleNamespace(count_tokens=claude_count))
    options: dict[str, Any] = {"rpm": 10, "max_retries": 1, "max_backoff_s": 60}
    gemini = GeminiClient("k", sdk=gemini_sdk, **options)
    claude = AnthropicClient("k", sdk=claude_sdk, **options)
    assert await gemini.count_tokens(model="gemini-3.8-flash", text="hello") == 12
    assert await claude.count_tokens(model="claude-sonnet-5-5", text="hello") == 9


# --- fallback model (free-tier Flash overloaded -> Flash-Lite) --------------------------------


def _gemini_fb(fake: FakeGemini, sleeps: Sleeps) -> GeminiClient:
    return GeminiClient(
        "k",
        sdk=fake,
        sleep=sleeps,
        rpm=1000,
        max_retries=1,
        max_backoff_s=60,
        fallback_model="gemini-3.5-flash-lite",
    )


async def test_overloaded_primary_falls_back_after_its_retries() -> None:
    overloaded = _gemini_error(503, "UNAVAILABLE")
    fake = FakeGemini(overloaded, overloaded, _ok(9))
    response = await _ask(_gemini_fb(fake, Sleeps()))
    assert response.parsed == Answer(value=9)
    assert (response.model, response.fallback_from) == ("gemini-3.5-flash-lite", "gemini-3.8-flash")
    assert [c["model"] for c in fake.calls] == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]


@pytest.mark.parametrize(
    ("error", "primary_calls"),
    [
        (_gemini_error(504, "DEADLINE_EXCEEDED"), 2),  # transient: 1 retry, then fallback
        # quota exhausted: no retry on the primary at all, straight to the fallback
        (_gemini_error(429, "RESOURCE_EXHAUSTED", [_retry_info("3600s")]), 1),
    ],
)
async def test_timeouts_and_exhausted_quota_also_fall_back(
    error: Exception, primary_calls: int
) -> None:
    fake = FakeGemini(*[error] * primary_calls, _ok())
    response = await _ask(_gemini_fb(fake, Sleeps()))
    assert response.fallback_from == "gemini-3.8-flash"
    assert [c["model"] for c in fake.calls].count("gemini-3.8-flash") == primary_calls


async def test_fatal_errors_never_fall_back() -> None:
    fake = FakeGemini(_gemini_error(400, "INVALID_ARGUMENT"), _ok())
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini_fb(fake, Sleeps()))
    assert raised.value.kind is ErrorKind.FATAL
    assert len(fake.calls) == 1


async def test_no_fallback_when_already_on_the_fallback_model() -> None:
    fake = FakeGemini(*[_gemini_error(503, "UNAVAILABLE")] * 4)
    with pytest.raises(LLMError):
        await _gemini_fb(fake, Sleeps()).structured(
            model="gemini-3.5-flash-lite", system="s", user="u", schema=Answer, temperature=0
        )
    assert {c["model"] for c in fake.calls} == {"gemini-3.5-flash-lite"}


async def test_fallback_failing_too_raises_its_error() -> None:
    fake = FakeGemini(*[_gemini_error(503, "UNAVAILABLE")] * 4)
    with pytest.raises(LLMError) as raised:
        await _ask(_gemini_fb(fake, Sleeps()))
    assert raised.value.kind is ErrorKind.RATE_LIMITED
    assert [c["model"] for c in fake.calls][-1] == "gemini-3.5-flash-lite"


async def test_success_on_primary_has_no_fallback_marker() -> None:
    response = await _ask(_gemini_fb(FakeGemini(_ok()), Sleeps()))
    assert (response.model, response.fallback_from) == ("gemini-3.8-flash", None)
