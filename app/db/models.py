"""App DB tables: query runs, guard events, eval runs and results."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class QueryRun(Base):
    __tablename__ = "query_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    lang: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    final_sql: Mapped[str | None] = mapped_column(Text)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    repairs_used: Mapped[int] = mapped_column(default=0, server_default="0")
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    confidence_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    judge_verdict: Mapped[str | None] = mapped_column(Text)
    judge_reason: Mapped[str | None] = mapped_column(Text)
    row_count: Mapped[int | None]
    confirmed_by_user: Mapped[bool] = mapped_column(default=False, server_default="false")
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    input_flags: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    gen_model: Mapped[str | None] = mapped_column(Text)
    judge_model: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(default=0, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), default=Decimal(0), server_default="0"
    )
    latency_ms: Mapped[int | None]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Ordered state history: [{state, at, detail}] (every transition).
    transitions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    # One entry per LLM call (generate / repair / judge): requested and answering model,
    # fallback, tokens, attempts, list-price cost, error.
    llm_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    # What the run would cost at paid list prices (cost_usd is the real cost: 0 on a free tier).
    cost_usd_list: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), default=Decimal(0), server_default="0"
    )

    guard_events: Mapped[list["GuardEvent"]] = relationship(
        back_populates="run", order_by="GuardEvent.id", passive_deletes=True
    )


Index("ix_query_runs_created_at", QueryRun.created_at.desc())
Index("ix_query_runs_status", QueryRun.status)


class GuardEvent(Base):
    __tablename__ = "guard_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("query_runs.id", ondelete="CASCADE"))
    stage: Mapped[str] = mapped_column(Text)
    rule: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool]
    severity: Mapped[str] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    sql_snapshot: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[QueryRun] = relationship(back_populates="guard_events")


Index("ix_guard_events_run_id", GuardEvent.run_id)
Index("ix_guard_events_rule_passed", GuardEvent.rule, GuardEvent.passed)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    git_sha: Mapped[str | None] = mapped_column(Text)
    gen_model: Mapped[str | None] = mapped_column(Text)
    judge_model: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(Text)
    suite: Mapped[str] = mapped_column(Text)  # golden / adversarial
    lang: Mapped[str | None] = mapped_column(Text)
    expected: Mapped[str | None] = mapped_column(Text)
    actual_status: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool | None]
    leaked: Mapped[bool] = mapped_column(default=False, server_default="false")
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("query_runs.id", ondelete="SET NULL")
    )
    latency_ms: Mapped[int | None]
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
