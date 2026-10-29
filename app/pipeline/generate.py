"""LLM SQL generation and repair (GENERATING).

The model only returns text, parsed into `SqlCandidate`; every candidate still goes through all
the guards (AST_CHECK -> SCHEMA_CHECK -> REWRITING -> DRY_RUN) before anything runs.
"""

from datetime import date

from pydantic import BaseModel, Field

from app.allowlist import Allowlist, load_allowlist
from app.pipeline.llm import LLMClient, LLMResponse
from app.pipeline.prompts import as_data, load_prompt
from app.pipeline.result import GuardResult, Outcome
from app.schema_context import build_schema_context

PRIMARY_TEMPERATURE = 0.0  # candidate #1
EXTRA_TEMPERATURE = 0.4  # candidates #2+ (self-consistency)


class SqlCandidate(BaseModel):
    answerable: bool = Field(description="false if this database cannot answer the question")
    sql: str | None = Field(default=None, description="one read-only PostgreSQL SELECT, or null")
    unanswerable_reason: str | None = Field(
        default=None, description="why it cannot be answered, when answerable is false"
    )
    tables_used: list[str] = Field(default_factory=list, description="tables the SQL reads")
    reasoning_short: str = Field(default="", description="one sentence on how the SQL answers it")


def render_generate_prompt(
    question: str, *, reference_date: date, allowlist: Allowlist | None = None
) -> tuple[str, str]:
    allowlist = allowlist or load_allowlist()
    return load_prompt("generate").render(
        schema=build_schema_context(allowlist).strip(),
        reference_date=reference_date.isoformat(),
        functions=", ".join(sorted(allowlist.functions)),
        question=as_data(question),
    )


def render_repair_prompt(
    question: str,
    previous_sql: str,
    failure: GuardResult,
    *,
    reference_date: date,
    allowlist: Allowlist | None = None,
) -> tuple[str, str]:
    """Generate's system prompt (same rules, one source) + the repair section; the question,
    previous SQL and exact error text go in the user message as tagged data (STATE.md rule 2)."""
    generate_system, _ = render_generate_prompt(
        question, reference_date=reference_date, allowlist=allowlist
    )
    repair_system, user = load_prompt("repair").render(
        question=as_data(question),
        previous_sql=as_data(previous_sql),
        error=as_data(f"{failure.stage} {failure.rule}: {failure.detail}"),
    )
    return f"{generate_system}\n\n{repair_system}", user


async def repair(
    question: str,
    previous_sql: str,
    failure: GuardResult,
    *,
    client: LLMClient,
    model: str,
    reference_date: date,
    allowlist: Allowlist | None = None,
) -> LLMResponse[SqlCandidate]:
    """Ask for a corrected query. Only repairable failures qualify: BLOCKED is never sent back
    to the model (STATE.md rule 1), and FAILED is terminal."""
    if failure.outcome is not Outcome.REPAIR:
        raise ValueError(
            f"only repairable results can be repaired, got {failure.outcome} ({failure.rule})"
        )
    system, user = render_repair_prompt(
        question, previous_sql, failure, reference_date=reference_date, allowlist=allowlist
    )
    return await client.structured(
        model=model, system=system, user=user, schema=SqlCandidate, temperature=PRIMARY_TEMPERATURE
    )


async def generate(
    question: str,
    *,
    client: LLMClient,
    model: str,
    reference_date: date,
    temperature: float = PRIMARY_TEMPERATURE,
    allowlist: Allowlist | None = None,
) -> LLMResponse[SqlCandidate]:
    system, user = render_generate_prompt(
        question, reference_date=reference_date, allowlist=allowlist
    )
    return await client.structured(
        model=model, system=system, user=user, schema=SqlCandidate, temperature=temperature
    )
