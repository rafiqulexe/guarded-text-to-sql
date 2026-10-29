"""Self-consistency: N candidates in parallel, every one through all the guards,
result-hash comparison of the survivors.

- Any BLOCKED candidate blocks the run (also a repaired one, and any permission error at execution).
- Candidate #1 decides `answerable`. Later candidates that say unanswerable, fail their LLM call,
  fail a guard repairably, or fail at execution are dropped (they count as non-agreeing).
- Only when no candidate survives the guards does the repair path run, on candidate #1
  (or the first repairable candidate), up to MAX_REPAIRS. Repairs only ever apply to the primary:
  a repairable DB error when the primary executes is repaired the same way; a primary timeout is
  FAILED.
- consistency = survivors whose result_hash equals the primary's / N.

`next_state` uses run state names: VALIDATING (go on), BLOCKED, FAILED, UNANSWERABLE.
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
from app.pipeline.result import GuardResult, Outcome, Stage

NextState = Literal["VALIDATING", "BLOCKED", "FAILED", "UNANSWERABLE"]
Dropped = Literal["unanswerable", "llm_error", "repair", "execution_failed"]
Executor = Callable[[str], execute.ExecResult]


# --- result hash -----------------------------------------------------------------


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


# --- hooks: how the runner sees each state, guard event and LLM call ----------------------------


@dataclass(frozen=True)
class Hooks:
    """Callbacks for the runner (persistence). Defaults do nothing."""

    on_state: Callable[[str, str], None] = lambda state, detail: None
    on_event: Callable[[GuardResult, "CandidateRecord", str], None] = lambda e, r, sql: None
    on_llm: Callable[["LLMCall"], None] = lambda call: None


@dataclass(frozen=True)
class LLMCall:
    """One generation or repair call: what was asked, which model answered, what it cost."""

    purpose: Literal["generate", "repair"]
    candidate: int
    requested_model: str
    response: LLMResponse[SqlCandidate] | None = None
    error: LLMError | None = None


NO_HOOKS = Hooks()

GUARD_STAGES = (Stage.AST_CHECK, Stage.SCHEMA_CHECK, Stage.REWRITING, Stage.DRY_RUN)


# --- candidates through the guards, one stage at a time -------------------------------------------


@dataclass
class GuardRun:
    """AST_CHECK -> SCHEMA_CHECK -> REWRITING -> DRY_RUN results for one candidate."""

    events: list[GuardResult]
    outcome: Outcome
    final_sql: str = ""
    limit_applied: bool = False

    @property
    def failure(self) -> GuardResult | None:
        return None if self.outcome is Outcome.PASSED else self.events[-1]


def _step(stage: Stage, value: Any, conninfo: str | None) -> tuple[GuardResult, Any]:
    """Run one guard stage; returns (result, input for the next stage)."""
    if stage is Stage.AST_CHECK:
        ast = ast_guard.check(value)
        return ast, ast.expression
    if stage is Stage.SCHEMA_CHECK:
        schema = schema_guard.check(value)
        return schema, schema.expression
    if stage is Stage.REWRITING:
        rewritten = rewrite.rewrite(value)
        return rewritten, rewritten
    plan = dry_run.dry_run(value.sql, conninfo=conninfo)
    return plan, value


def _guard_stagewise(
    records: list["CandidateRecord"],
    conninfo: str | None,
    events: list[GuardResult],
    hooks: Hooks,
) -> GuardResult | None:
    """All candidates through each guard stage in turn (so the run's state moves forward only).
    Returns the BLOCKED event as soon as any candidate is blocked, else None."""
    work: dict[int, Any] = {}
    for record in records:
        record.guards = GuardRun([], Outcome.PASSED)
        work[id(record)] = record.sql or ""  # empty SQL -> AST_CHECK empty_sql (repair)
    for stage in GUARD_STAGES:
        live = [r for r in records if r.guards is not None and r.guards.outcome is Outcome.PASSED]
        if not live:
            break
        hooks.on_state(stage, f"{len(live)} candidate(s)")
        for record in live:
            assert record.guards is not None
            result, next_value = _step(stage, work[id(record)], conninfo)
            snapshot = work[id(record)].sql if stage is Stage.DRY_RUN else (record.sql or "")
            if stage is Stage.REWRITING and isinstance(result, rewrite.RewriteResult):
                snapshot = result.sql or snapshot
            record.guards.events.append(result)
            events.append(result)
            hooks.on_event(result, record, snapshot)
            if result.outcome is Outcome.BLOCKED:
                record.guards.outcome = Outcome.BLOCKED
                return result
            if not result.passed or next_value is None:
                record.guards.outcome = result.outcome
                continue
            work[id(record)] = next_value
    for record in records:
        assert record.guards is not None
        if record.guards.outcome is Outcome.PASSED:
            rewritten = work[id(record)]
            assert isinstance(rewritten, rewrite.RewriteResult)
            record.guards.final_sql, record.guards.limit_applied = (
                rewritten.sql,
                rewritten.limit_applied,
            )
            record.final_sql, record.limit_applied = rewritten.sql, rewritten.limit_applied
            record.limit = rewritten.limit
    return None


def guard_sql(sql: str, *, conninfo: str | None = None) -> GuardRun:
    """Every guard for one SQL string (e.g. for scripts)."""
    record = CandidateRecord(
        1, PRIMARY_TEMPERATURE, candidate=SqlCandidate(answerable=True, sql=sql)
    )
    _guard_stagewise([record], conninfo, [], NO_HOOKS)
    assert record.guards is not None
    return record.guards


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
    limit: int = 0  # outer LIMIT after REWRITING
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
    hooks: Hooks = NO_HOOKS,
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

    # 1. GENERATING: N candidates in parallel.
    hooks.on_state("GENERATING", f"{n} candidate(s)")
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
            hooks.on_llm(LLMCall("generate", record.index, model, error=response))
        elif isinstance(response, BaseException):
            raise response
        else:
            _take(record, response, out)
            hooks.on_llm(LLMCall("generate", record.index, model, response=response))

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
    checked = [r for r in answered if r.dropped is None]
    blocking = _guard_stagewise(checked, conninfo, out.events, hooks)
    if blocking is not None:
        where = next(r.index for r in checked if r.guards and r.guards.outcome is Outcome.BLOCKED)
        return _block(out, blocking, f"candidate #{where}")
    for record in checked:
        if record.guards and record.guards.outcome is not Outcome.PASSED:
            record.dropped = (
                "repair" if record.guards.outcome is Outcome.REPAIR else "execution_failed"
            )
    survivors = [r for r in checked if r.guards and r.guards.outcome is Outcome.PASSED]

    # 4. nothing survived: repair candidate #1 (or the first repairable one).
    if not survivors:
        target = next((r for r in checked if r.guards and r.guards.outcome is Outcome.REPAIR), None)
        if target is None:
            return _end(out, "FAILED", "no candidate survived the guards")
        repaired = await _repair_loop(
            question,
            target,
            client,
            model,
            reference_date,
            allowlist,
            conninfo,
            max_repairs,
            out,
            hooks=hooks,
        )
        if repaired is None:
            return out
        survivors = [repaired]

    # 5. EXECUTING every survivor; the primary is #1 if it survived, else the first survivor.
    primary = survivors[0]
    hooks.on_state("EXECUTING", f"{len(survivors)} survivor(s)")
    executions = await asyncio.gather(
        *(asyncio.to_thread(run_exec, r.final_sql) for r in survivors)
    )
    for record, result in zip(survivors, executions, strict=True):
        record.execution = result
        out.events.append(result)
        hooks.on_event(result, record, record.final_sql)
    for record in survivors:
        assert record.execution is not None
        if record.execution.outcome is Outcome.BLOCKED:
            return _block(out, record.execution, f"candidate #{record.index} at execution")
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
            hooks=hooks,
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
    hooks: Hooks = NO_HOOKS,
) -> CandidateRecord | None:
    """Repair `target` until a version passes the guards (and, if `run_exec`, executes).
    Returns the new record, or None after setting out.next_state (BLOCKED / FAILED / UNANSWERABLE).
    """
    current = target
    while out.repairs_used < max_repairs:
        failure = _failure_of(current)
        previous_sql = current.sql or ""
        out.repairs_used += 1
        record = CandidateRecord(current.index, PRIMARY_TEMPERATURE, repaired_from=previous_sql)
        out.candidates.append(record)
        hooks.on_state("GENERATING", f"repair {out.repairs_used}/{max_repairs} of #{record.index}")
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
            hooks.on_llm(LLMCall("repair", record.index, model, error=exc))
            _end(out, "FAILED", f"repair call failed: {exc}")
            return None
        _take(record, response, out)
        hooks.on_llm(LLMCall("repair", record.index, model, response=response))
        assert record.candidate is not None
        if not record.candidate.answerable:
            out.unanswerable_reason = record.candidate.unanswerable_reason
            _end(out, "UNANSWERABLE", f"after repair: {out.unanswerable_reason}")
            return None
        blocking = _guard_stagewise([record], conninfo, out.events, hooks)
        assert record.guards is not None
        if blocking is not None:
            _block(out, blocking, f"repair of candidate #{record.index}")
            return None
        if record.guards.outcome is Outcome.PASSED:
            if run_exec is None:
                return record
            hooks.on_state("EXECUTING", f"repaired candidate #{record.index}")
            result = await asyncio.to_thread(run_exec, record.final_sql)
            record.execution = result
            out.events.append(result)
            hooks.on_event(result, record, record.final_sql)
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


def _block(out: ConsistencyResult, event: GuardResult, where: str) -> ConsistencyResult:
    out.blocking = event
    return _end(out, "BLOCKED", f"{where}: {event.stage} {event.rule}: {event.detail}")


def _end(out: ConsistencyResult, state: NextState, detail: str) -> ConsistencyResult:
    out.next_state, out.detail = state, detail
    return out
