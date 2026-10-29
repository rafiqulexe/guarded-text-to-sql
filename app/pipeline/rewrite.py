"""REWRITING: emit the checked, *-expanded tree as SQL with a bounded outer LIMIT.

Only the outermost query is limited; subquery LIMITs and ORDER BY are kept. Any LIMIT that
can't be read as a plain integer (ALL, NULL, an expression, FETCH ... WITH TIES) is replaced
or wrapped with MAX_LIMIT. The emitted SQL is re-checked by AST_CHECK before it leaves here.
"""

from dataclasses import dataclass

from sqlglot import exp

from app.config import get_guard_settings
from app.pipeline import ast_guard
from app.pipeline.result import GuardResult, Outcome, Severity, Stage

DIALECT = "postgres"
STAGE = Stage.REWRITING


@dataclass(frozen=True)
class RewriteResult(GuardResult):
    sql: str = ""
    limit: int = 0
    limit_injected: bool = False  # we added a LIMIT (DEFAULT_LIMIT)
    limit_clamped: bool = False  # we lowered or replaced the model's LIMIT (MAX_LIMIT)

    @property
    def limit_applied(self) -> bool:
        """A LIMIT we set, so hitting it says nothing about the model (STATE.md §4 sanity)."""
        return self.limit_injected or self.limit_clamped


def rewrite(
    tree: exp.Expr, *, default_limit: int | None = None, max_limit: int | None = None
) -> RewriteResult:
    if default_limit is None or max_limit is None:
        settings = get_guard_settings()
        default_limit = settings.default_limit if default_limit is None else default_limit
        max_limit = settings.max_limit if max_limit is None else max_limit
    if not 0 < default_limit <= max_limit:
        raise ValueError(f"need 0 < default_limit ({default_limit}) <= max_limit ({max_limit})")

    query = tree.copy()
    while isinstance(query, exp.Subquery):
        query = query.this
    if not isinstance(query, exp.Query):
        return _blocked(f"cannot limit a {type(query).__name__}")

    limit, injected, clamped = _bound_limit(query, default_limit, max_limit)
    if clamped and isinstance(limit, exp.Query):
        query, limit = limit, max_limit  # FETCH ... WITH TIES: wrapped in an outer query

    sql = query.sql(dialect=DIALECT, comments=False)
    recheck = ast_guard.check(sql)
    if not recheck.passed:
        return _blocked(f"rewritten SQL failed AST_CHECK ({recheck.rule}): {recheck.detail}")

    assert isinstance(limit, int)
    detail = f"LIMIT {limit}" + (" injected" if injected else " clamped" if clamped else " kept")
    return RewriteResult(
        STAGE,
        Outcome.PASSED,
        "limit_injected" if injected else "limit_clamped" if clamped else "limit_kept",
        detail,
        Severity.INFO,
        query,
        sql=sql,
        limit=limit,
        limit_injected=injected,
        limit_clamped=clamped,
    )


def _bound_limit(
    query: exp.Query, default_limit: int, max_limit: int
) -> tuple[int | exp.Query, bool, bool]:
    """Set the outer LIMIT in place. Returns (limit or wrapping query, injected, clamped)."""
    limit = query.args.get("limit")  # sqlglot stores FETCH FIRST here too
    if isinstance(limit, exp.Fetch):
        options = limit.args.get("limit_options")
        if options is not None and (options.args.get("with_ties") or options.args.get("percent")):
            wrapped = exp.select("*").from_(query.subquery("limited")).limit(max_limit)
            return wrapped, False, True
        count = limit.args.get("count")
        requested = 1 if count is None else _int_literal(count)
        query.set("limit", None)
        return _set_limit(query, requested, max_limit)

    if limit is None:
        query.limit(default_limit, copy=False)
        return default_limit, True, False
    return _set_limit(query, _int_literal(limit.expression), max_limit)


def _set_limit(query: exp.Query, requested: int | None, max_limit: int) -> tuple[int, bool, bool]:
    if requested is not None and requested <= max_limit:
        query.limit(requested, copy=False)
        return requested, False, False
    query.limit(max_limit, copy=False)
    return max_limit, False, True


def _int_literal(node: exp.Expr | None) -> int | None:
    """A plain non-negative integer literal, else None (treated as unbounded)."""
    if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit():
        return int(node.this)
    return None


def _blocked(detail: str) -> RewriteResult:
    return RewriteResult(STAGE, Outcome.BLOCKED, "rewrite_recheck", detail, Severity.BLOCK)
