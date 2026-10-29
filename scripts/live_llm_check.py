"""Real-API checks. Run on purpose; needs the provider key in .env and uses free-tier quota.

    uv run python scripts/live_llm_check.py client   # one structured-output call

Unit tests never call an LLM (the client is mocked); this script is the only place that does.
"""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.pipeline.llm import make_client


class Ping(BaseModel):
    answer: int
    language: str


async def check_client() -> None:
    settings = get_settings()
    client = make_client(settings)
    response = await client.structured(
        model=settings.gen_model,
        system="Return JSON matching the schema.",
        user="What is 6 * 7? Also name the language of this sentence: 'আমি ভাত খাই'.",
        schema=Ping,
        temperature=0,
    )
    print(
        f"provider={client.provider} model={response.model} attempts={response.attempts}"
        f" fallback_from={response.fallback_from}"
    )
    print(f"parsed={response.parsed!r}")
    print(f"tokens in={response.input_tokens} out={response.output_tokens}")


CHECKS: dict[str, Callable[[], Awaitable[None]]] = {"client": check_client}


def main() -> None:
    names = sys.argv[1:] or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        sys.exit(f"unknown check(s) {unknown}; choose from {list(CHECKS)}")
    for name in names:
        print(f"=== {name} ===")
        asyncio.run(CHECKS[name]())


if __name__ == "__main__":
    main()
