"""State machine for a query run (STATE.md), persisted to the app DB.

Every transition updates `query_runs.status`, `updated_at` and the ordered `transitions` history;
every guard result writes a `guard_events` row (one per candidate per guard stage). Transitions
are checked against STATE.md §2: a terminal state can never be left, so e.g. BLOCKED can never
re-enter GENERATING.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.allowlist import Allowlist, load_allowlist
from app.config import Settings, get_settings
from app.db.models import GuardEvent, QueryRun
from app.db.session import get_sessionmaker
from app.pipeline import dry_run as dry_run_guard
from app.pipeline import execute as execute_guard
from app.pipeline.consistency import (
    CandidateRecord,
    ConsistencyResult,
    Executor,
    Hooks,
    LLMCall,
    run_self_consistency,
)
from app.pipeline.llm import LLMClient, LLMError
from app.pipeline.result import GuardResult, Outcome, Severity, Stage, blocked, passed
from app.pipeline.score import ScoreInputs, ScoreResult, make_scorer
from app.pipeline.validate import JudgeFn, make_judge

TERMINAL = frozenset({"SUCCEEDED", "BLOCKED", "FAILED", "CANCELLED", "UNANSWERABLE"})

# STATE.md §2 "Exits to". FAILED ("unrecoverable error") is also allowed from every
# non-terminal state, so an unexpected exception can always end a run cleanly.
_EXITS: dict[str, set[str]] = {
    "RECEIVED": {"INPUT_CHECK"},
    "INPUT_CHECK": {"GENERATING", "BLOCKED"},
    "GENERATING": {"AST_CHECK", "UNANSWERABLE"},
    "AST_CHECK": {"SCHEMA_CHECK", "GENERATING", "BLOCKED"},
    "SCHEMA_CHECK": {"REWRITING", "GENERATING", "BLOCKED"},
    "REWRITING": {"DRY_RUN", "BLOCKED"},
    "DRY_RUN": {"EXECUTING", "GENERATING", "BLOCKED"},
    "EXECUTING": {"VALIDATING", "GENERATING", "BLOCKED"},
    "VALIDATING": {"SCORING"},
    "SCORING": {"SUCCEEDED", "AWAITING_CONFIRMATION"},
    "AWAITING_CONFIRMATION": {"SUCCEEDED", "CANCELLED", "BLOCKED"},
}
ALLOWED: dict[str, frozenset[str]] = {
    state: frozenset(exits | {"FAILED"}) for state, exits in _EXITS.items()
} | {state: frozenset() for state in TERMINAL}

JUDGE_PREVIEW_ROWS = 20

# INPUT_CHECK: obvious prompt-injection / SQL-in-the-question markers. They only *flag*
# (`possible_injection`): the run continues but can never auto-SUCCEED (STATE.md rule 9).
INJECTION_MARKERS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|above)\b",
        r"\b(system|developer|hidden)\s+(prompt|message|instructions?)\b",
        r"\byou are now\b|\bact as\b|\bpretend (to be|you are)\b|\bjailbreak\b",
        r"</?\s*(question|system|schema|instructions?|previous_sql|error|rows)\s*>",
        r";\s*(drop|delete|update|insert|alter|truncate|grant|create)\b",
        r"\b(drop|truncate|alter)\s+table\b|\bdelete\s+from\b|\binsert\s+into\b",
        r"\b(pg_sleep\w*|pg_read_file|dblink|pg_shadow|pg_catalog|information_schema)\b",
        r"(নির্দেশ|নিয়ম).{0,30}(উপেক্ষা|ভুলে)",  # Bangla: ignore / forget the instructions / rules
    )
]
_BANGLA = re.compile(r"[ঀ-৿]")

ScoreFn = Callable[[ScoreInputs], ScoreResult]


class IllegalTransition(RuntimeError):
    pass


@dataclass
class RunOutcome:
    """What the API returns. Rows only for SUCCEEDED (never stored, STATE.md rule 4)."""

    run_id: uuid.UUID
    status: str
    detail: str = ""
    sql: str | None = None
    confidence: float | None = None
    breakdown: dict[str, float] | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    input_flags: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    guard_events: list[dict[str, Any]] = field(default_factory=list)


def check_transition(current: str, new: str) -> None:
    if new not in ALLOWED.get(current, frozenset()):
        raise IllegalTransition(f"{current} -> {new} is not allowed by STATE.md")


def input_check(question: str, max_chars: int) -> tuple[GuardResult, list[str], str]:
    """(event, input_flags, lang). Empty / too long -> BLOCKED; injection markers only flag."""
    lang = "bn" if _BANGLA.search(question) else "en"
    text = question.strip()
    if not text:
        return blocked(Stage.INPUT_CHECK, "empty_question", "the question is empty"), [], lang
    if len(text) > max_chars:
        detail = f"question has {len(text)} characters; the limit is {max_chars}"
        return blocked(Stage.INPUT_CHECK, "question_too_long", detail), [], lang
    hits = [m.pattern for m in INJECTION_MARKERS if m.search(text)]
    if hits:
        event = GuardResult(
            Stage.INPUT_CHECK,
            Outcome.PASSED,
            "possible_injection",
            f"{len(hits)} injection marker(s) matched; results will need confirmation",
            Severity.WARN,
        )
        return event, ["possible_injection"], lang
    return passed(Stage.INPUT_CHECK, "input_ok", f"{len(text)} characters, lang={lang}"), [], lang


class _Run:
    """One run's in-memory state + its DB row; every change is written through immediately."""

    def __init__(self, sessions: sessionmaker[Session], run_id: uuid.UUID) -> None:
        self.sessions, self.id = sessions, run_id
        self.state = "RECEIVED"
        self.started = time.monotonic()
        self.transitions: list[str] = ["RECEIVED"]
        self.events: list[dict[str, Any]] = []

    @classmethod
    def create(cls, sessions: sessionmaker[Session], question: str, **fields: Any) -> "_Run":
        run_id = uuid.uuid4()
        now = datetime.now(UTC)
        with sessions.begin() as s:
            s.add(
                QueryRun(
                    id=run_id,
                    question=question,
                    status="RECEIVED",
                    transitions=[{"state": "RECEIVED", "at": now.isoformat(), "detail": ""}],
                    **fields,
                )
            )
        return cls(sessions, run_id)

    @classmethod
    def load(cls, sessions: sessionmaker[Session], run_id: uuid.UUID) -> tuple["_Run", QueryRun]:
        with sessions() as s:
            row = s.get(QueryRun, run_id)
            if row is None:
                raise KeyError(f"run {run_id} not found")
            s.expunge(row)
        run = cls(sessions, run_id)
        run.state = row.status
        run.transitions = [t["state"] for t in row.transitions]
        return run, row

    def to(self, state: str, detail: str = "", **fields: Any) -> None:
        check_transition(self.state, state)
        now = datetime.now(UTC)
        with self.sessions.begin() as s:
            row = s.get(QueryRun, self.id)
            assert row is not None
            row.status = state
            row.updated_at = now
            row.transitions = [
                *row.transitions,
                {"state": state, "at": now.isoformat(), "detail": detail[:500]},
            ]
            for name, value in fields.items():
                setattr(row, name, value)
            if state in TERMINAL:
                row.finished_at = now
                row.latency_ms = int((now - row.created_at).total_seconds() * 1000)
        self.state = state
        self.transitions.append(state)

    def update(self, **fields: Any) -> None:
        with self.sessions.begin() as s:
            row = s.get(QueryRun, self.id)
            assert row is not None
            for name, value in fields.items():
                setattr(row, name, value)

    def event(self, result: GuardResult, sql: str = "", candidate: int | None = None) -> None:
        where = f"[#{candidate}] " if candidate is not None else ""
        record = {
            "stage": str(result.stage),
            "rule": result.rule,
            "passed": result.passed,
            "severity": str(result.severity),
            "detail": f"{where}{result.detail}",
            "sql_snapshot": sql or None,
        }
        with self.sessions.begin() as s:
            s.add(GuardEvent(run_id=self.id, **record))
        self.events.append(record)


class Runner:
    def __init__(
        self,
        *,
        client: LLMClient,
        scorer: ScoreFn | None = None,
        judge: JudgeFn | None = None,
        settings: Settings | None = None,
        sessions: sessionmaker[Session] | None = None,
        allowlist: Allowlist | None = None,
        conninfo: str | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self.scorer: ScoreFn = scorer or make_scorer(self.settings)
        self.judge: JudgeFn = judge or make_judge(client, self.settings.judge_model)
        self.sessions = sessions or get_sessionmaker()
        self.allowlist = allowlist or load_allowlist()
        self.conninfo = conninfo
        self.executor: Executor = executor or (
            lambda sql: execute_guard.execute(sql, conninfo=self.conninfo)
        )

    # --- POST /api/ask ------------------------------------------------------------------------

    async def ask(self, question: str) -> RunOutcome:
        settings = self.settings
        run = _Run.create(
            self.sessions,
            question,
            gen_model=settings.gen_model,
            judge_model=settings.judge_model,
        )
        try:
            return await self._ask(run, question)
        except IllegalTransition:
            raise
        except Exception as exc:  # unrecoverable: end the run FAILED, never leave it hanging
            if run.state not in TERMINAL:
                run.to("FAILED", f"unexpected error: {exc}", error=f"{type(exc).__name__}: {exc}")
            return self._outcome(run, "FAILED", f"unexpected error: {exc}")

    async def _ask(self, run: _Run, question: str) -> RunOutcome:
        settings = self.settings
        # INPUT_CHECK
        run.to("INPUT_CHECK")
        event, flags, lang = input_check(question, settings.max_question_chars)
        run.event(event)
        run.update(input_flags=flags, lang=lang)
        if event.outcome is Outcome.BLOCKED:
            run.to("BLOCKED", event.detail, error=event.detail)
            return self._outcome(run, "BLOCKED", event.detail, flags=flags)

        # GENERATING -> AST_CHECK -> SCHEMA_CHECK -> REWRITING -> DRY_RUN -> EXECUTING (+ repairs)
        assert settings.reference_date is not None, "REFERENCE_DATE must be set"
        result = await run_self_consistency(
            question,
            client=self.client,
            model=settings.gen_model,
            reference_date=settings.reference_date,
            n=settings.self_consistency_n,
            max_repairs=settings.max_repairs,
            allowlist=self.allowlist,
            conninfo=self.conninfo,
            executor=self.executor,
            hooks=self._hooks(run),
        )
        common = self._consistency_fields(result)
        if result.next_state != "VALIDATING":
            run.to(result.next_state, result.detail, error=result.detail, **common)
            return self._outcome(
                run, result.next_state, result.detail, flags=flags, sql=common.get("final_sql")
            )

        primary = result.primary
        assert primary is not None and primary.execution is not None
        # VALIDATING
        run.to("VALIDATING", f"judging candidate #{primary.index}", **common)
        try:
            judged = await self.judge(
                question, primary.final_sql, result.columns, result.rows[:JUDGE_PREVIEW_ROWS]
            )
        except LLMError as exc:
            detail = f"judge call failed: {exc}"
            run.to("FAILED", detail, error=detail)
            return self._outcome(run, "FAILED", detail, flags=flags, sql=primary.final_sql)
        verdict = judged.parsed
        run.event(
            GuardResult(
                Stage.VALIDATING,
                Outcome.PASSED,
                f"judge_{verdict.verdict}",
                f"verdict={verdict.verdict} expects_rows={verdict.expects_rows}: {verdict.reason}",
                Severity.INFO if verdict.verdict == "yes" else Severity.WARN,
            ),
            primary.final_sql,
            primary.index,
        )

        # SCORING
        score = self.scorer(
            ScoreInputs(
                verdict=verdict.verdict,
                expects_rows=verdict.expects_rows,
                consistency=result.consistency,
                repairs_used=result.repairs_used,
                max_repairs=settings.max_repairs,
                row_count=len(result.rows),
                limit=primary.limit,
                limit_applied=primary.limit_applied,
            )
        )
        scored = {
            "confidence": Decimal(str(round(score.score, 3))),
            "confidence_breakdown": score.breakdown,
            "judge_verdict": verdict.verdict,
            "judge_reason": verdict.reason,
            "judge_model": judged.model,
            "input_tokens": result.input_tokens + judged.input_tokens,
            "output_tokens": result.output_tokens + judged.output_tokens,
        }
        run.to("SCORING", f"score={score.score:.3f}", **scored)
        needs_confirmation = (
            "possible_injection" in flags or score.score < settings.confidence_threshold
        )
        if needs_confirmation:
            why = (
                "possible_injection flag"
                if "possible_injection" in flags
                else f"score {score.score:.3f} < {settings.confidence_threshold}"
            )
            run.to("AWAITING_CONFIRMATION", why)  # rows are not stored or returned (rule 4)
            return self._outcome(
                run, "AWAITING_CONFIRMATION", why, flags=flags, sql=primary.final_sql, score=score
            )
        run.to("SUCCEEDED", f"score={score.score:.3f}")
        return self._outcome(
            run,
            "SUCCEEDED",
            f"score={score.score:.3f}",
            flags=flags,
            sql=primary.final_sql,
            score=score,
            columns=result.columns,
            rows=result.rows,
        )

    # --- POST /api/runs/{id}/confirm --------------------------------------------------------

    async def confirm(self, run_id: uuid.UUID) -> RunOutcome:
        """Re-run the *stored* final_sql through DRY_RUN -> EXECUTING (same guards, same role).
        Never regenerates: no LLM call is made."""
        run, row = _Run.load(self.sessions, run_id)
        check_transition(run.state, "SUCCEEDED")  # only from AWAITING_CONFIRMATION
        sql = row.final_sql or ""
        plan = dry_run_guard.dry_run(sql, conninfo=self.conninfo)
        run.event(plan, sql)
        if not plan.passed:
            state = "BLOCKED" if plan.outcome is Outcome.BLOCKED else "FAILED"
            run.to(state, f"on confirm: {plan.rule}: {plan.detail}", error=plan.detail)
            return self._outcome(run, state, plan.detail, sql=sql)
        executed = await _in_thread(self.executor, sql)
        run.event(executed, sql)
        if not executed.passed:
            state = "BLOCKED" if executed.outcome is Outcome.BLOCKED else "FAILED"
            run.to(state, f"on confirm: {executed.rule}: {executed.detail}", error=executed.detail)
            return self._outcome(run, state, executed.detail, sql=sql)
        run.to(
            "SUCCEEDED",
            "confirmed by user; final_sql re-executed",
            confirmed_by_user=True,
            confirmed_at=datetime.now(UTC),
            row_count=len(executed.rows),
        )
        return RunOutcome(
            run.id,
            "SUCCEEDED",
            "confirmed by user",
            sql=sql,
            confidence=float(row.confidence) if row.confidence is not None else None,
            breakdown=row.confidence_breakdown,
            columns=executed.columns,
            rows=executed.rows,
            input_flags=list(row.input_flags),
            transitions=run.transitions,
            guard_events=run.events,
        )

    # --- POST /api/runs/{id}/cancel ---------------------------------------------------------

    def cancel(self, run_id: uuid.UUID) -> RunOutcome:
        run, row = _Run.load(self.sessions, run_id)
        run.to("CANCELLED", "cancelled by user")
        return RunOutcome(
            run.id, "CANCELLED", "cancelled by user", sql=row.final_sql, transitions=run.transitions
        )

    # --- helpers -------------------------------------------------------------------------------

    def _hooks(self, run: _Run) -> Hooks:
        def on_state(state: str, detail: str) -> None:
            if state != run.state:
                run.to(state, detail)

        def on_event(result: GuardResult, record: CandidateRecord, sql: str) -> None:
            run.event(result, sql, record.index)

        def on_llm(call: LLMCall) -> None:  # per-call accounting: 4.4
            return None

        return Hooks(on_state=on_state, on_event=on_event, on_llm=on_llm)

    @staticmethod
    def _consistency_fields(result: ConsistencyResult) -> dict[str, Any]:
        primary = result.primary
        fields: dict[str, Any] = {
            "candidates": result.candidates_log(),
            "repairs_used": result.repairs_used,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        if primary is not None:
            fields["final_sql"] = primary.final_sql
            fields["row_count"] = len(result.rows)
            if primary.model:
                fields["gen_model"] = primary.model  # the model that actually answered
        return fields

    @staticmethod
    def _outcome(
        run: _Run,
        status: str,
        detail: str,
        *,
        flags: list[str] | None = None,
        sql: str | None = None,
        score: ScoreResult | None = None,
        columns: tuple[str, ...] = (),
        rows: tuple[tuple[Any, ...], ...] = (),
    ) -> RunOutcome:
        return RunOutcome(
            run.id,
            status,
            detail,
            sql=sql,
            confidence=score.score if score else None,
            breakdown=score.breakdown if score else None,
            columns=columns,
            rows=rows,
            input_flags=flags or [],
            transitions=run.transitions,
            guard_events=run.events,
        )


async def _in_thread(executor: Executor, sql: str) -> execute_guard.ExecResult:
    return await asyncio.to_thread(executor, sql)
