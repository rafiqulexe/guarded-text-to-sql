"""Judge LLM validation (VALIDATING): does the SQL + a preview of its rows answer the question?

The question, the SQL and every row value are untrusted data: they go in tags in the user
message, tag-like text in them is neutralised, and long values are truncated.

The judge is told the real date range of the data and "today" for it (from config), so it judges
dates against this database and not against what it remembers about public sample datasets.
"""

import datetime as dt
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.allowlist import Allowlist, load_allowlist
from app.config import Settings
from app.pipeline.llm import LLMClient, LLMResponse
from app.pipeline.prompts import as_data, load_prompt
from app.schema_context import build_schema_context

PREVIEW_ROWS = 20  # the judge sees the first 20 rows
MAX_VALUE_CHARS = 200
JUDGE_TEMPERATURE = 0.0


@dataclass(frozen=True)
class DataDates:
    """First and last activity date in the data, and "today" for it (REFERENCE_DATE)."""

    start: dt.date
    end: dt.date
    reference: dt.date

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataDates":
        start, end, ref = settings.data_start_date, settings.data_end_date, settings.reference_date
        if start is None or end is None or ref is None:
            raise ValueError("DATA_START_DATE, DATA_END_DATE and REFERENCE_DATE must be set")
        return cls(start, end, ref)


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
    dates: DataDates,
    allowlist: Allowlist | None = None,
) -> tuple[str, str]:
    shown = rows[:PREVIEW_ROWS]
    return load_prompt("judge").render(
        schema=build_schema_context(allowlist or load_allowlist()).strip(),
        data_start=dates.start.isoformat(),
        data_end=dates.end.isoformat(),
        reference_date=dates.reference.isoformat(),
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
    dates: DataDates,
    allowlist: Allowlist | None = None,
) -> LLMResponse[JudgeVerdict]:
    system, user = render_judge_prompt(
        question, sql, columns, rows, dates=dates, allowlist=allowlist
    )
    return await client.structured(
        model=model, system=system, user=user, schema=JudgeVerdict, temperature=JUDGE_TEMPERATURE
    )


def make_judge(client: LLMClient, model: str, dates: DataDates) -> "JudgeFn":
    """The runner's JudgeFn: (question, sql, columns, rows) -> verdict."""

    async def judge_fn(
        question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> LLMResponse[JudgeVerdict]:
        return await judge(question, sql, columns, rows, client=client, model=model, dates=dates)

    return judge_fn


JudgeFn = Callable[
    [str, str, Sequence[str], Sequence[Sequence[Any]]], Awaitable[LLMResponse[JudgeVerdict]]
]
