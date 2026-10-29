"""Judge LLM validation (VALIDATING): does the SQL + a preview of its rows answer the question?

The question, the SQL and every row value are untrusted data: they go in tags in the user
message, tag-like text in them is neutralised, and long values are truncated.
"""

import datetime as dt
import json
from collections.abc import Awaitable, Callable, Sequence
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.allowlist import Allowlist, load_allowlist
from app.pipeline.llm import LLMClient, LLMResponse
from app.pipeline.prompts import as_data, load_prompt
from app.schema_context import build_schema_context

PREVIEW_ROWS = 20  # PLAN 4.2: first 20 rows
MAX_VALUE_CHARS = 200
JUDGE_TEMPERATURE = 0.0


class JudgeVerdict(BaseModel):
    verdict: Literal["yes", "partial", "no"] = Field(
        description="does the SQL and its result answer the question?"
    )
    expects_rows: bool = Field(description="would a correct answer return at least one row?")
    reason: str = Field(description="one or two sentences explaining the verdict")


def _value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Decimal):
        return f"{value:f}"
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_value(v) for v in value]
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(bytes(value))} bytes>"
    text = str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "…"


def _rows_block(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [json.dumps({"columns": [str(c) for c in columns]}, ensure_ascii=False)]
    lines += [json.dumps([_value(v) for v in row], ensure_ascii=False) for row in rows]
    return as_data("\n".join(lines))


def render_judge_prompt(
    question: str,
    sql: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    allowlist: Allowlist | None = None,
) -> tuple[str, str]:
    shown = rows[:PREVIEW_ROWS]
    return load_prompt("judge").render(
        schema=build_schema_context(allowlist or load_allowlist()).strip(),
        shown_limit=str(PREVIEW_ROWS),
        question=as_data(question),
        sql=as_data(sql),
        row_count=str(len(rows)),
        shown=str(len(shown)),
        rows=_rows_block(columns, shown),
    )


async def judge(
    question: str,
    sql: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    client: LLMClient,
    model: str,
    allowlist: Allowlist | None = None,
) -> LLMResponse[JudgeVerdict]:
    system, user = render_judge_prompt(question, sql, columns, rows, allowlist=allowlist)
    return await client.structured(
        model=model, system=system, user=user, schema=JudgeVerdict, temperature=JUDGE_TEMPERATURE
    )


def make_judge(client: LLMClient, model: str) -> "JudgeFn":
    """The runner's JudgeFn: (question, sql, columns, rows) -> verdict."""

    async def judge_fn(
        question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> LLMResponse[JudgeVerdict]:
        return await judge(question, sql, columns, rows, client=client, model=model)

    return judge_fn


JudgeFn = Callable[
    [str, str, Sequence[str], Sequence[Sequence[Any]]], Awaitable[LLMResponse[JudgeVerdict]]
]
