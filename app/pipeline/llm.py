"""Provider-neutral LLM client: one structured-output call, per-model rate limiting, retries.

generate / repair / validate depend only on `LLMClient.structured()`; which provider answers is
config (LLM_PROVIDER). The model only ever returns text parsed into a Pydantic model; it never
touches the database.

Retry policy (STATE.md GENERATING / VALIDATING -> FAILED):
- rate limited (429) or overloaded: retried up to LLM_MAX_RETRIES with exponential backoff,
  or the server's retry delay when it gives one;
- other transient errors (5xx, network, unparseable output): 1 retry;
- quota exhausted (retry delay > LLM_MAX_BACKOFF_S, e.g. the free-tier daily quota) and
  client errors (bad key, bad request): no retry.
"""

import asyncio
import random
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

import anthropic
import httpx
import httpx2
import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from app.config import LLMSettings, get_llm_settings

log = structlog.get_logger(__name__)

TRANSIENT_RETRIES = 1  # STATE.md: API error after 1 retry -> FAILED
BACKOFF_INITIAL_S = 2.0
MAX_OUTPUT_TOKENS = 8192  # thinking models spend part of this before answering


class ErrorKind(StrEnum):
    RATE_LIMITED = "rate_limited"  # 429 / overloaded: wait and retry
    TRANSIENT = "transient"  # 5xx, network, bad output: retry once
    QUOTA_EXHAUSTED = "quota_exhausted"  # e.g. daily free-tier quota: give up now
    FATAL = "fatal"  # auth, bad request: give up now


class LLMError(Exception):
    def __init__(self, message: str, kind: ErrorKind, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


@dataclass(frozen=True)
class LLMResponse[T: BaseModel]:
    parsed: T
    model: str
    input_tokens: int
    output_tokens: int
    attempts: int = 1
    fallback_from: str | None = None  # set when the primary model failed and this one answered


class LLMClient(Protocol):
    provider: str

    async def structured[T: BaseModel](
        self, *, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]: ...

    async def count_tokens(self, *, model: str, text: str) -> int: ...


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class SlidingWindowLimiter:
    """At most `rpm` call starts in any 60 s window. A burst (e.g. N parallel candidates)
    goes through at once while the window has room."""

    def __init__(self, rpm: int, *, clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep):
        self._rpm, self._clock, self._sleep = rpm, clock, sleep
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()
                if len(self._starts) < self._rpm:
                    self._starts.append(now)
                    return
                await self._sleep(60 - (now - self._starts[0]))


class RetryingClient(ABC):
    """Rate limiting + retry around one provider call (`_call_once`)."""

    provider = "base"

    def __init__(
        self,
        *,
        rpm: int,
        max_retries: int,
        max_backoff_s: float,
        fallback_model: str = "",
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self._rpm, self._max_retries, self._max_backoff = rpm, max_retries, max_backoff_s
        self._fallback = fallback_model
        self._sleep, self._clock = sleep, clock
        self._limiters: dict[str, SlidingWindowLimiter] = {}

    async def structured[T: BaseModel](
        self, *, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]:
        """Call `model` with retries; if it still fails for a non-fatal reason (overloaded,
        rate limited, quota exhausted, transient) and a different fallback model is set, make
        the same request once more on the fallback model (with its own retries)."""
        try:
            return await self._with_retries(model, system, user, schema, temperature)
        except LLMError as exc:
            if exc.kind is ErrorKind.FATAL or not self._fallback or self._fallback == model:
                raise
            log.warning("llm_fallback", model=model, fallback=self._fallback, kind=exc.kind.value)
            response = await self._with_retries(self._fallback, system, user, schema, temperature)
            return replace(response, fallback_from=model)

    async def _with_retries[T: BaseModel](
        self, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]:
        limiter = self._limiters.setdefault(
            model, SlidingWindowLimiter(self._rpm, clock=self._clock, sleep=self._sleep)
        )
        rate_limited = transient = 0
        while True:
            await limiter.acquire()
            try:
                result = await self._call_once(
                    model=model, system=system, user=user, schema=schema, temperature=temperature
                )
                attempts = 1 + rate_limited + transient
                return LLMResponse(
                    result.parsed, result.model, result.input_tokens, result.output_tokens, attempts
                )
            except LLMError as exc:
                if exc.kind is ErrorKind.RATE_LIMITED:
                    if exc.retry_after is not None and exc.retry_after > self._max_backoff:
                        raise LLMError(
                            f"quota exhausted (retry after {exc.retry_after:.0f}s): {exc}",
                            ErrorKind.QUOTA_EXHAUSTED,
                            exc.retry_after,
                        ) from exc
                    rate_limited += 1
                    if rate_limited > self._max_retries:
                        raise
                    await self._wait(
                        model, exc, rate_limited, self._backoff(rate_limited, exc.retry_after)
                    )
                elif exc.kind is ErrorKind.TRANSIENT:
                    transient += 1
                    if transient > TRANSIENT_RETRIES:
                        raise
                    await self._wait(
                        model, exc, transient, self._backoff(transient, exc.retry_after)
                    )
                else:
                    raise

    async def _wait(self, model: str, exc: LLMError, retry: int, seconds: float) -> None:
        log.warning(
            "llm_retry",
            model=model,
            kind=exc.kind.value,
            retry=retry,
            wait_s=round(seconds, 1),
            error=str(exc)[:200],
        )
        await self._sleep(seconds)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._max_backoff)
        exponential = BACKOFF_INITIAL_S * 2 ** (attempt - 1)
        return float(min(exponential, self._max_backoff) * random.uniform(0.8, 1.2))

    @abstractmethod
    async def _call_once[T: BaseModel](
        self, *, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]: ...


# --- Gemini (google-genai) --------------------------------------------------------------

# google-genai runs on httpx or httpx2 (aiohttp is an optional extra we do not install).
_NETWORK_ERRORS = (httpx.TransportError, httpx2.TransportError)


class GeminiClient(RetryingClient):
    provider = "gemini"

    def __init__(
        self, api_key: str, *, sdk: Any = None, timeout_s: float = 60, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # timeout in ms; attempts=1: the SDK must not retry, our RetryingClient is the only layer
        http_options = genai_types.HttpOptions(
            timeout=int(timeout_s * 1000), retry_options=genai_types.HttpRetryOptions(attempts=1)
        )
        self._sdk = (
            sdk if sdk is not None else genai.Client(api_key=api_key, http_options=http_options)
        )

    async def count_tokens(self, *, model: str, text: str) -> int:
        try:
            result = await self._sdk.aio.models.count_tokens(model=model, contents=text)
        except genai_errors.APIError as exc:
            raise _gemini_error(exc) from exc
        return int(result.total_tokens or 0)

    async def _call_once[T: BaseModel](
        self, *, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            response = await self._sdk.aio.models.generate_content(
                model=model, contents=user, config=config
            )
        except genai_errors.APIError as exc:
            raise _gemini_error(exc) from exc
        except _NETWORK_ERRORS as exc:
            raise LLMError(f"network error: {exc}", ErrorKind.TRANSIENT) from exc

        parsed = response.parsed if isinstance(response.parsed, schema) else None
        if parsed is None:
            try:
                parsed = schema.model_validate_json(response.text or "")
            except ValidationError as exc:
                raise LLMError(
                    f"output did not match {schema.__name__}: {exc.errors()[:1]}",
                    ErrorKind.TRANSIENT,
                ) from exc
        usage = response.usage_metadata
        if usage is None:
            return LLMResponse(parsed, model, 0, 0)
        output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return LLMResponse(parsed, model, usage.prompt_token_count or 0, output)


def _gemini_error(exc: genai_errors.APIError) -> LLMError:
    message = f"gemini {exc.code} {exc.status}: {exc.message}"
    if exc.code == 429:
        details = _error_details(exc.details)
        if any(
            "PerDay" in str(v.get("quotaId", "")) for d in details for v in d.get("violations", [])
        ):
            return LLMError(f"daily quota exhausted: {message}", ErrorKind.QUOTA_EXHAUSTED)
        return LLMError(message, ErrorKind.RATE_LIMITED, _gemini_retry_delay(details))
    if exc.code in (500, 502, 503, 504):
        kind = (
            ErrorKind.RATE_LIMITED if exc.code == 503 else ErrorKind.TRANSIENT
        )  # 503 = overloaded
        return LLMError(message, kind)
    return LLMError(message, ErrorKind.FATAL)


def _error_details(payload: Any) -> list[dict[str, Any]]:
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    details = error.get("details", []) if isinstance(error, dict) else []
    return [d for d in details if isinstance(d, dict)]


def _gemini_retry_delay(details: list[dict[str, Any]]) -> float | None:
    for detail in details:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None


# --- Anthropic ---------------------------------------------------------------------------


class AnthropicClient(RetryingClient):
    """Structured output via `messages.parse(output_format=...)`.

    `temperature` is accepted for interface parity but not sent: current Claude models
    (Sonnet 5.5, Haiku 5.5) reject sampling parameters, so self-consistency candidates differ
    only by the model's default sampling. Forced tool choice is also rejected, hence parse().
    """

    provider = "anthropic"

    def __init__(
        self, api_key: str, *, sdk: Any = None, timeout_s: float = 60, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Retries are ours (shared policy for both providers), so the SDK must not retry too.
        self._sdk = (
            sdk
            if sdk is not None
            else anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_s)
        )

    async def count_tokens(self, *, model: str, text: str) -> int:
        result = await self._sdk.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        )
        return int(result.input_tokens)

    async def _call_once[T: BaseModel](
        self, *, model: str, system: str, user: str, schema: type[T], temperature: float
    ) -> LLMResponse[T]:
        try:
            message = await self._sdk.messages.parse(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(str(exc), ErrorKind.RATE_LIMITED, _retry_after(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 529:  # overloaded
                raise LLMError(str(exc), ErrorKind.RATE_LIMITED, _retry_after(exc)) from exc
            kind = ErrorKind.TRANSIENT if exc.status_code >= 500 else ErrorKind.FATAL
            raise LLMError(str(exc), kind) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"network error: {exc}", ErrorKind.TRANSIENT) from exc

        if message.stop_reason == "refusal":
            raise LLMError("model refused the request", ErrorKind.FATAL)
        parsed = message.parsed_output
        if not isinstance(parsed, schema):
            raise LLMError(f"output did not match {schema.__name__}", ErrorKind.TRANSIENT)
        return LLMResponse(parsed, model, message.usage.input_tokens, message.usage.output_tokens)


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


# --- factory --------------------------------------------------------------------------------


def make_client(settings: LLMSettings | None = None) -> LLMClient:
    settings = settings or get_llm_settings()
    assert settings.api_key is not None  # LLMSettings validates this
    options: dict[str, Any] = {
        "rpm": settings.llm_rpm,
        "max_retries": settings.llm_max_retries,
        "max_backoff_s": settings.llm_max_backoff_s,
        "timeout_s": settings.llm_timeout_s,
        "fallback_model": settings.llm_fallback_model,
    }
    key = settings.api_key.get_secret_value()
    if settings.llm_provider == "gemini":
        return GeminiClient(key, **options)
    return AnthropicClient(key, **options)
