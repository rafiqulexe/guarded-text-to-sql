"""Self-consistency (STATE.md rule 8): N candidates in parallel, every one through all the guards,
result-hash comparison of the survivors.

- Any BLOCKED candidate blocks the run (also a repaired one, and any permission error at execution).
- Candidate #1 decides `answerable`. Later candidates that say unanswerable, fail their LLM call,
  fail a guard repairably, or fail at execution are dropped (they count as non-agreeing).
- Only when no candidate survives the guards does the repair path run, on candidate #1
  (or the first repairable candidate), up to MAX_REPAIRS. Repairs only ever apply to the primary:
  a repairable DB error when the primary executes is repaired the same way; a primary timeout is
  FAILED (STATE.md rule 3).
- consistency = survivors whose result_hash equals the primary's / N.

`next_state` uses STATE.md names: VALIDATING (go on), BLOCKED, FAILED, UNANSWERABLE.
"""

import asyncio
import datetime as dt
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from app.allowlist import Allowlist, load_allowlist
from app.pipeline import ast_guard, dry_run, execute, rewrite, schema_guard
from app.pipeline.generate import (
    EXTRA_TEMPERATURE,
    PRIMARY_TEMPERATURE,
    SqlCandidate,
    generate,
)
from app.pipeline.generate import repair as repair_sql
from app.pipeline.llm import LLMClient, LLMError, LLMResponse
from app.pipeline.result import GuardResult, Outcome

NextState = Literal["VALIDATING", "BLOCKED", "FAILED", "UNANSWERABLE"]
Dropped = Literal["unanswerable", "llm_error", "repair", "execution_failed"]
Executor = Callable[[str], execute.ExecResult]


# --- result hash (SCHEMA.md §4) -----------------------------------------------------------------


def _normalise(value: Any) -> Any:
    """Type-tagged, JSON-able form: numbers rounded to 2 dp, timestamps ISO (UTC)."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int | float | Decimal):
        return ["num", f"{Decimal(str(value)).quantize(Decimal('0.01')):f}"]
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC)
        return ["ts", value.isoformat()]
    if isinstance(value, dt.date | dt.time):
        return ["date", value.isoformat()]
    if isinstance(value, list | tuple):
        return ["arr", [_normalise(v) for v in value]]
    return ["str", str(value)]


def result_hash(rows: Sequence[Sequence[Any]], *, order_insensitive_cols: bool = False) -> str:
    """SHA-256 of the sorted, normalised rows. Column names are never part of it; column order
    is, unless `order_insensitive_cols`."""
    normalised = []
    for row in rows:
        values = [json.dumps(_normalise(v), sort_keys=True) for v in row]
        if order_insensitive_cols:
            values.sort()
        normalised.append(values)
    normalised.sort()
    return hashlib.sha256(json.dumps(normalised).encode()).hexdigest()


# --- one candidate through the guards -------------------------------------------------------------


@dataclass
class GuardRun:
    """AST_CHECK -> SCHEMA_CHECK -> REWRITING -> DRY_RUN for one SQL string."""

    events: list[GuardResult]
    outcome: Outcome
    final_sql: str = ""
    limit_applied: bool = False

    @property
    def failure(self) -> GuardResult | None:
        return None if self.outcome is Outcome.PASSED else self.events[-1]


def guard_sql(sql: str, *, conninfo: str | None = None) -> GuardRun:
    events: list[GuardResult] = []
    ast = ast_guard.check(sql)
    events.append(ast)
    if not ast.passed or ast.expression is None:
        return GuardRun(events, ast.outcome)
    schema = schema_guard.check(ast.expression)
    events.append(schema)
    if not schema.passed or schema.expression is None:
        return GuardRun(events, schema.outcome)
    rewritten = rewrite.rewrite(schema.expression)
    events.append(rewritten)
    if not rewritten.passed:
        return GuardRun(events, rewritten.outcome)
    plan = dry_run.dry_run(rewritten.sql, conninfo=conninfo)
    events.append(plan)
    if not plan.passed:
        return GuardRun(events, plan.outcome)
    return GuardRun(events, Outcome.PASSED, rewritten.sql, rewritten.limit_applied)


# --- records --------------------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    index: int  # 1-based; #1 is the temperature-0 primary
    temperature: float
    model: str | None = None
    fallback_from: str | None = None
    candidate: SqlCandidate | None = None
    llm_error: str | None = None
    guards: GuardRun | None = None
    final_sql: str = ""
    limit_applied: bool = False
    result_hash: str | None = None
    execution: execute.ExecResult | None = None
    dropped: Dropped | None = None
    repaired_from: str | None = None  # previous SQL when this record is a repair

    @property
    def sql(self) -> str | None:
        return self.candidate.sql if self.candidate else None

    @property
    def error(self) -> str | None:
        if self.llm_error:
            return self.llm_error
        failure = self.guards.failure if self.guards else None
        if failure:
            return f"{failure.stage} {failure.rule}: {failure.detail}"
        if self.execution is not None and not self.execution.passed:
            return f"{self.execution.stage} {self.execution.rule}: {self.execution.detail}"
        return None


@dataclass
class ConsistencyResult:
    next_state: NextState
    detail: str
    candidates: list[CandidateRecord]
    primary: CandidateRecord | None = None
    consistency: float = 0.0
    repairs_used: int = 0
    blocking: GuardResult | None = None  # the event that blocked the run
    unanswerable_reason: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[GuardResult] = field(default_factory=list)

    def candidates_log(self) -> list[dict[str, Any]]:
        """`query_runs.candidates` entries: {sql, result_hash, error} (+ context)."""
        return [
            {
                "index": c.index,
                "model": c.model,
                "fallback_from": c.fallback_from,
                "sql": c.sql,
                "final_sql": c.final_sql or None,
                "result_hash": c.result_hash,
                "error": c.error,
                "dropped": c.dropped,
                "repaired_from": c.repaired_from,
            }
            for c in self.candidates
        ]


# --- the run --------------------------------------------------------------------------------------


async def run_self_consistency(
    question: str,
    *,
    client: LLMClient,
    model: str,
    reference_date: dt.date,
    n: int,
    max_repairs: int,
    allowlist: Allowlist | None = None,
    conninfo: str | None = None,
    executor: Executor | None = None,
) -> ConsistencyResult:
    if n < 1:
        raise ValueError("need at least one candidate")
    allowlist = allowlist or load_allowlist()
    run_exec: Executor = executor or (lambda sql: execute.execute(sql, conninfo=conninfo))
    records = [
        CandidateRecord(i + 1, PRIMARY_TEMPERATURE if i == 0 else EXTRA_TEMPERATURE)
        for i in range(n)
    ]
    out = ConsistencyResult("FAILED", "", records)

    # 1. N candidates in parallel.
    responses = await asyncio.gather(
        *(
            generate(
                question,
                client=client,
                model=model,
                reference_date=reference_date,
                temperature=r.temperature,
                allowlist=allowlist,
            )
            for r in records
        ),
        return_exceptions=True,
    )
    for record, response in zip(records, responses, strict=True):
        if isinstance(response, LLMError):
            record.llm_error, record.dropped = f"LLM: {response}", "llm_error"
        elif isinstance(response, BaseException):
            raise response
        else:
            _take(record, response, out)

    # 2. answerable: candidate #1 decides (or the first candidate whose call returned).
    answered = [r for r in records if r.candidate is not None]
    if not answered:
        return _end(out, "FAILED", "every LLM call failed")
    decider = answered[0]
    assert decider.candidate is not None
    if not decider.candidate.answerable:
        out.unanswerable_reason = decider.candidate.unanswerable_reason
        return _end(out, "UNANSWERABLE", f"candidate #{decider.index}: {out.unanswerable_reason}")
    for record in answered:
        if record is not decider and record.candidate and not record.candidate.answerable:
            record.dropped = "unanswerable"

    # 3. every remaining candidate through every guard; any BLOCKED blocks the run.
    for record in answered:
        if record.dropped is None:
            _guard(record, conninfo, out)
    blocked_record = next(
        (r for r in records if r.guards and r.guards.outcome is Outcome.BLOCKED), None
    )
    if blocked_record:
        assert blocked_record.guards is not None
        return _block(out, blocked_record.guards.events[-1], f"candidate #{blocked_record.index}")
    for record in records:
        if record.guards and record.guards.outcome is not Outcome.PASSED and record.dropped is None:
            record.dropped = (
                "repair" if record.guards.outcome is Outcome.REPAIR else "execution_failed"
            )

    survivors = [r for r in records if r.guards and r.guards.outcome is Outcome.PASSED]

    # 4. nothing survived: repair candidate #1 (or the first repairable one).
    if not survivors:
        target = next((r for r in records if r.guards and r.guards.outcome is Outcome.REPAIR), None)
        if target is None:
            return _end(out, "FAILED", "no candidate survived the guards")
        repaired = await _repair_loop(
            question, target, client, model, reference_date, allowlist, conninfo, max_repairs, out
        )
        if repaired is None:
            return out
        survivors = [repaired]

    # 5. execute every survivor; the primary is #1 if it survived, else the first survivor.
    primary = survivors[0]
    executions = await asyncio.gather(
        *(asyncio.to_thread(run_exec, r.final_sql) for r in survivors)
    )
    for record, result in zip(survivors, executions, strict=True):
        record.execution = result
        out.events.append(result)
        if result.outcome is Outcome.BLOCKED:
            return _block(out, result, f"candidate #{record.index} at execution")
    for record in survivors:
        ran = record.execution
        assert ran is not None
        if ran.passed:
            record.result_hash = result_hash(ran.rows)
        elif record is not primary:
            record.dropped = "execution_failed"

    primary_exec = primary.execution
    assert primary_exec is not None
    if primary_exec.outcome is Outcome.FAILED:
        return _end(out, "FAILED", f"primary: {primary_exec.rule}: {primary_exec.detail}")
    if primary_exec.outcome is Outcome.REPAIR:
        primary.dropped = "execution_failed"
        repaired = await _repair_loop(
            question,
            primary,
            client,
            model,
            reference_date,
            allowlist,
            conninfo,
            max_repairs,
            out,
            run_exec=run_exec,
        )
        if repaired is None:
            return out
        primary = repaired

    # 6. consistency against the primary's result.
    assert primary.execution is not None and primary.result_hash is not None
    agreeing = sum(1 for r in records if r.result_hash == primary.result_hash and r.dropped is None)
    out.primary, out.consistency = primary, agreeing / n
    out.columns, out.rows = primary.execution.columns, primary.execution.rows
    return _end(
        out, "VALIDATING", f"{agreeing}/{n} candidates agree with candidate #{primary.index}"
    )


async def _repair_loop(
    question: str,
    target: CandidateRecord,
    client: LLMClient,
    model: str,
    reference_date: dt.date,
    allowlist: Allowlist,
    conninfo: str | None,
    max_repairs: int,
    out: ConsistencyResult,
    *,
    run_exec: Executor | None = None,
) -> CandidateRecord | None:
    """Repair `target` until a version passes the guards (and, if `run_exec`, executes).
    Returns the new record, or None after setting out.next_state (BLOCKED / FAILED)."""
    current = target
    while out.repairs_used < max_repairs:
        failure = _failure_of(current)
        previous_sql = current.sql or ""
        out.repairs_used += 1
        record = CandidateRecord(current.index, PRIMARY_TEMPERATURE, repaired_from=previous_sql)
        out.candidates.append(record)
        try:
            response = await repair_sql(
                question,
                previous_sql,
                failure,
                client=client,
                model=model,
                reference_date=reference_date,
                allowlist=allowlist,
            )
        except LLMError as exc:
            record.llm_error = f"LLM: {exc}"
            _end(out, "FAILED", f"repair call failed: {exc}")
            return None
        _take(record, response, out)
        assert record.candidate is not None
        if not record.candidate.answerable:
            out.unanswerable_reason = record.candidate.unanswerable_reason
            _end(out, "UNANSWERABLE", f"after repair: {out.unanswerable_reason}")
            return None
        _guard(record, conninfo, out)
        assert record.guards is not None
        if record.guards.outcome is Outcome.BLOCKED:
            _block(out, record.guards.events[-1], f"repair of candidate #{record.index}")
            return None
        if record.guards.outcome is Outcome.PASSED:
            if run_exec is None:
                return record
            result = await asyncio.to_thread(run_exec, record.final_sql)
            record.execution = result
            out.events.append(result)
            if result.passed:
                record.result_hash = result_hash(result.rows)
                return record
            if result.outcome is Outcome.BLOCKED:
                _block(out, result, f"repair of candidate #{record.index} at execution")
                return None
            if result.outcome is Outcome.FAILED:
                _end(out, "FAILED", f"repaired primary: {result.rule}: {result.detail}")
                return None
        elif record.guards.outcome is Outcome.FAILED:
            _end(out, "FAILED", f"repair of candidate #{record.index}: {record.error}")
            return None
        current = record
    _end(out, "FAILED", f"repairs exhausted ({max_repairs})")
    return None


def _failure_of(record: CandidateRecord) -> GuardResult:
    if record.guards and record.guards.failure:
        return record.guards.failure
    assert record.execution is not None
    return record.execution


def _take(
    record: CandidateRecord, response: LLMResponse[SqlCandidate], out: ConsistencyResult
) -> None:
    record.candidate, record.model = response.parsed, response.model
    record.fallback_from = response.fallback_from
    out.input_tokens += response.input_tokens
    out.output_tokens += response.output_tokens


def _guard(record: CandidateRecord, conninfo: str | None, out: ConsistencyResult) -> None:
    run = guard_sql(
        record.sql or "", conninfo=conninfo
    )  # empty SQL -> AST_CHECK empty_sql (repair)
    record.guards, record.final_sql, record.limit_applied = run, run.final_sql, run.limit_applied
    out.events.extend(run.events)


def _block(out: ConsistencyResult, event: GuardResult, where: str) -> ConsistencyResult:
    out.blocking = event
    return _end(out, "BLOCKED", f"{where}: {event.stage} {event.rule}: {event.detail}")


def _end(out: ConsistencyResult, state: NextState, detail: str) -> ConsistencyResult:
    out.next_state, out.detail = state, detail
    return out
