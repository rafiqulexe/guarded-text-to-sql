"""FastAPI app and routes (STACK.md §6).

Every route goes through the pipeline `Runner` (STATE.md); nothing here touches the target DB
except /health's connectivity probe. Rows are only ever returned by ask (SUCCEEDED) and confirm:
they are never stored (STATE.md rule 4), so GET /api/runs/{id} has none.
"""

import datetime as dt
import math
import uuid
from collections.abc import Awaitable, Callable
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.db.models import QueryRun
from app.pipeline.llm import make_client
from app.pipeline.runner import IllegalTransition, Runner, RunOutcome

settings = get_settings()

app = FastAPI(title="sqlguard-llm")

APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

HEALTH_CONNECT_TIMEOUT_S = 3
MAX_LIST_LIMIT = 200

# Everything the UI shows (question, SQL, rows, error text) is untrusted. The UI only ever sets
# textContent; this CSP is the second layer: no inline script, nothing from other origins.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


# --- request / response models ------------------------------------------------------------------


class AskRequest(BaseModel):
    # Length is enforced by INPUT_CHECK (empty / > MAX_QUESTION_CHARS -> BLOCKED, and logged).
    question: str


class Transition(BaseModel):
    state: str
    at: str
    detail: str = ""


class GuardEventOut(BaseModel):
    stage: str
    rule: str
    passed: bool
    severity: str
    detail: str | None
    sql_snapshot: str | None


class LLMCallOut(BaseModel):
    purpose: str
    candidate: int | None
    requested_model: str
    model: str | None  # the model that actually answered
    fallback_from: str | None
    input_tokens: int
    output_tokens: int
    error: str | None


class RunSummary(BaseModel):
    run_id: uuid.UUID
    question: str
    status: str
    confidence: float | None
    created_at: dt.datetime
    latency_ms: int | None


class RunDetail(BaseModel):
    """A stored run: everything in query_runs + its guard_events. Never rows."""

    run_id: uuid.UUID
    question: str
    lang: str | None
    status: str
    detail: str  # detail of the latest transition
    error: str | None
    sql: str | None  # final SQL, or the SQL a guard stopped (always shown, CLAUDE.md rule 5)
    confidence: float | None
    breakdown: dict[str, float] | None
    judge_verdict: str | None
    judge_reason: str | None
    input_flags: list[str]
    repairs_used: int
    row_count: int | None
    confirmed_by_user: bool
    gen_model: str | None  # the model that answered (after any fallback)
    judge_model: str | None
    llm_calls: list[LLMCallOut]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_usd_list: float
    created_at: dt.datetime
    finished_at: dt.datetime | None
    latency_ms: int | None
    transitions: list[Transition]
    guard_events: list[GuardEventOut]


class RunResult(RunDetail):
    """ask / confirm: the stored run plus the rows of this request (SUCCEEDED only)."""

    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    limit: int | None = None  # outer LIMIT, for "showing N of LIMIT"


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    target_db: Literal["ok", "unreachable"]
    app_db: Literal["ok", "unreachable"]
    llm_provider: str
    api_key_configured: bool  # presence only: the key itself is never returned or logged
    gen_model: str
    judge_model: str


# --- dependencies ---------------------------------------------------------------------------------


@lru_cache
def get_runner() -> Runner:
    return Runner(client=make_client(settings), settings=settings)


RunnerDep = Annotated[Runner, Depends(get_runner)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --- UI -------------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request, current: SettingsDep) -> HTMLResponse:
    # Badge cutoffs and limits come from config (TOKENS.md), rendered as JSON data, never code.
    config = {
        "confidenceThreshold": current.confidence_threshold,
        "confidenceHigh": current.confidence_high,
        "maxQuestionChars": current.max_question_chars,
        "weights": {
            "judge": current.score_weight_judge,
            "consistency": current.score_weight_consistency,
            "repairs": current.score_weight_repairs,
            "sanity": current.score_weight_sanity,
        },
    }
    return templates.TemplateResponse(request, "index.html", {"config": config})


# --- API ------------------------------------------------------------------------------------------


@app.post("/api/ask")
async def ask(body: AskRequest, runner: RunnerDep) -> RunResult:
    outcome = await runner.ask(body.question)
    return _result(runner, outcome)


@app.post("/api/runs/{run_id}/confirm", responses={404: {}, 409: {}})
async def confirm(run_id: uuid.UUID, runner: RunnerDep) -> RunResult:
    _detail(runner, run_id)  # 404 before anything else
    try:
        outcome = await runner.confirm(run_id)
    except IllegalTransition as exc:
        raise HTTPException(409, f"run cannot be confirmed: {exc}") from exc
    return _result(runner, outcome)


@app.post("/api/runs/{run_id}/cancel", responses={404: {}, 409: {}})
def cancel(run_id: uuid.UUID, runner: RunnerDep) -> RunDetail:
    _detail(runner, run_id)
    try:
        runner.cancel(run_id)
    except IllegalTransition as exc:
        raise HTTPException(409, f"run cannot be cancelled: {exc}") from exc
    return _detail(runner, run_id)


@app.get("/api/runs/{run_id}", responses={404: {}})
def get_run(run_id: uuid.UUID, runner: RunnerDep) -> RunDetail:
    return _detail(runner, run_id)


@app.get("/api/runs")
def list_runs(
    runner: RunnerDep, limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = 50
) -> list[RunSummary]:
    with runner.sessions() as s:
        rows = s.scalars(select(QueryRun).order_by(QueryRun.created_at.desc()).limit(limit))
        return [
            RunSummary(
                run_id=r.id,
                question=r.question,
                status=r.status,
                confidence=_float(r.confidence),
                created_at=r.created_at,
                latency_ms=r.latency_ms,
            )
            for r in rows
        ]


@app.get("/health", responses={503: {"model": HealthOut}})
def health(current: SettingsDep) -> JSONResponse:
    """DB reachability + whether an API key is configured. Exception text is never returned:
    a driver message can carry connection details."""
    target = _reachable(current.target_db_url)
    app_db = _reachable(current.app_db_url)
    key = current.api_key
    has_key = key is not None and bool(key.get_secret_value().strip())
    ok = target and app_db and has_key
    body = HealthOut(
        status="ok" if ok else "degraded",
        target_db="ok" if target else "unreachable",
        app_db="ok" if app_db else "unreachable",
        llm_provider=current.llm_provider,
        api_key_configured=has_key,
        gen_model=current.gen_model,
        judge_model=current.judge_model,
    )
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


# --- helpers --------------------------------------------------------------------------------------


def _reachable(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=HEALTH_CONNECT_TIMEOUT_S) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return False
    return True


def _detail(runner: Runner, run_id: uuid.UUID) -> RunDetail:
    with runner.sessions() as s:
        row = s.scalar(
            select(QueryRun)
            .where(QueryRun.id == run_id)
            .options(selectinload(QueryRun.guard_events))
        )
        if row is None:
            raise HTTPException(404, f"run {run_id} not found")
        events = [
            GuardEventOut(
                stage=e.stage,
                rule=e.rule,
                passed=e.passed,
                severity=e.severity,
                detail=e.detail,
                sql_snapshot=e.sql_snapshot,
            )
            for e in row.guard_events
        ]
        transitions = [Transition.model_validate(t) for t in row.transitions]
        return RunDetail(
            run_id=row.id,
            question=row.question,
            lang=row.lang,
            status=row.status,
            detail=transitions[-1].detail if transitions else "",
            error=row.error,
            sql=row.final_sql or _stopped_sql(events, row.candidates),
            confidence=_float(row.confidence),
            breakdown=row.confidence_breakdown,
            judge_verdict=row.judge_verdict,
            judge_reason=row.judge_reason,
            input_flags=list(row.input_flags),
            repairs_used=row.repairs_used,
            row_count=row.row_count,
            confirmed_by_user=row.confirmed_by_user,
            gen_model=row.gen_model,
            judge_model=row.judge_model,
            llm_calls=[LLMCallOut.model_validate(c) for c in row.llm_calls],
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cost_usd=float(row.cost_usd),
            cost_usd_list=float(row.cost_usd_list),
            created_at=row.created_at,
            finished_at=row.finished_at,
            latency_ms=row.latency_ms,
            transitions=transitions,
            guard_events=events,
        )


def _stopped_sql(events: list[GuardEventOut], candidates: list[dict[str, Any]]) -> str | None:
    """No final SQL (e.g. BLOCKED at AST_CHECK): show the SQL the failing guard saw."""
    for event in events:
        if not event.passed and event.sql_snapshot:
            return event.sql_snapshot
    for candidate in candidates:
        if candidate.get("sql"):
            return str(candidate["sql"])
    return None


def _result(runner: Runner, outcome: RunOutcome) -> RunResult:
    detail = _detail(runner, outcome.run_id)
    if outcome.status != "SUCCEEDED":  # rows only ever leave with SUCCEEDED
        return RunResult(**detail.model_dump())
    return RunResult(
        **detail.model_dump(),
        columns=list(outcome.columns),
        rows=[[_cell(v) for v in row] for row in outcome.rows],
        limit=outcome.limit,
    )


def _cell(value: Any) -> Any:
    """A DB value as plain JSON. Decimals stay exact (as strings); the UI treats them as text."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, bytes | memoryview):
        return "\\x" + bytes(value).hex()
    if isinstance(value, list | tuple):
        return [_cell(v) for v in value]
    return str(value)


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)
