"""DRY_RUN (layer 3): EXPLAIN (FORMAT JSON) as sqlguard_reader.

BLOCKED if any plan node exceeds MAX_COST or MAX_PLAN_ROWS. Every node is checked, not just the
root: an injected LIMIT makes a huge cross join look cheap at the top (`Limit cost=1 rows=100`).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg

from app.config import get_settings
from app.pipeline.execute import classify_db_error, reader_transaction, run_guarded
from app.pipeline.result import GuardResult, Outcome, Severity, Stage

STAGE = Stage.DRY_RUN


@dataclass(frozen=True)
class DryRunResult(GuardResult):
    max_cost: float = 0.0  # highest "Total Cost" of any plan node
    max_plan_rows: float = 0.0  # highest "Plan Rows" of any plan node
    worst_node: str = ""


def dry_run(
    sql: str,
    *,
    conninfo: str | None = None,
    max_cost: float | None = None,
    max_plan_rows: float | None = None,
    timeout_ms: int | None = None,
) -> DryRunResult:
    settings = get_settings()
    conninfo = conninfo or settings.target_db_url
    max_cost = settings.max_cost if max_cost is None else max_cost
    max_plan_rows = settings.max_plan_rows if max_plan_rows is None else max_plan_rows
    timeout_ms = settings.statement_timeout_ms if timeout_ms is None else timeout_ms

    try:
        with reader_transaction(conninfo, timeout_ms) as cur:
            run_guarded(cur, f"EXPLAIN (FORMAT JSON) {sql}")
            row = cur.fetchone()
    except psycopg.Error as exc:
        return _from(classify_db_error(STAGE, exc))
    if row is None:
        return _result(Outcome.FAILED, "explain_empty", "EXPLAIN returned no plan", Severity.WARN)

    nodes = list(_walk(row[0][0]["Plan"]))
    costliest = max(nodes, key=lambda n: n["Total Cost"])
    widest = max(nodes, key=lambda n: n["Plan Rows"])
    cost, rows = float(costliest["Total Cost"]), float(widest["Plan Rows"])
    stats = {"max_cost": cost, "max_plan_rows": rows}

    if cost > max_cost:
        return _result(
            Outcome.BLOCKED,
            "max_cost",
            f"{costliest['Node Type']} estimated cost {cost:,.0f} > MAX_COST {max_cost:,.0f}",
            Severity.BLOCK,
            worst_node=costliest["Node Type"],
            **stats,
        )
    if rows > max_plan_rows:
        return _result(
            Outcome.BLOCKED,
            "max_plan_rows",
            f"{widest['Node Type']} estimated {rows:,.0f} rows"
            f" > MAX_PLAN_ROWS {max_plan_rows:,.0f}",
            Severity.BLOCK,
            worst_node=widest["Node Type"],
            **stats,
        )
    return _result(
        Outcome.PASSED,
        "plan_ok",
        f"max node cost {cost:,.0f}, max node rows {rows:,.0f}",
        Severity.INFO,
        **stats,
    )


def _walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


def _result(
    outcome: Outcome, rule: str, detail: str, severity: Severity, **fields: Any
) -> DryRunResult:
    return DryRunResult(STAGE, outcome, rule, detail, severity, **fields)


def _from(result: GuardResult) -> DryRunResult:
    return _result(result.outcome, result.rule, result.detail, result.severity)
