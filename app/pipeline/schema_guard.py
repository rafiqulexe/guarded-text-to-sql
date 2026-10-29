"""SCHEMA_CHECK (layer 2): hallucination check against the allowlist.

Two qualify passes over a copy of the AST_CHECK tree:
1. against allowed + restricted columns, with `*` expanded, so every column the query *wrote*
   can be traced through aliases, subqueries and CTEs to its base table. A written column that
   reaches a restricted column is BLOCKED; so is a whole-row reference (`SELECT c FROM customer c`),
   which would read every column.
2. against allowed columns only: unknown tables/columns go back for repair, and `*` expands to
   allowlisted columns. The returned tree is what REWRITING emits.
"""

from typing import Any

from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.allowlist import Allowlist, load_allowlist
from app.pipeline.result import GuardResult, Stage, blocked, passed, repair

DIALECT = "postgres"
STAGE = Stage.SCHEMA_CHECK
_WRITTEN = "sqlguard_written"  # meta flag: column was in the SQL, not added by * expansion


def check(tree: exp.Expr, allowlist: Allowlist | None = None) -> GuardResult:
    allowlist = allowlist or load_allowlist()
    restricted = allowlist.restricted_columns()

    # Pass 1: trace written columns to their base tables.
    traced = tree.copy()
    for column in traced.find_all(exp.Column):
        column.meta[_WRITTEN] = True
    try:
        traced = _qualify(traced, _schema(allowlist, with_restricted=True), validate=False)
        scopes = traverse_scope(traced)
    except OptimizeError as exc:
        # A qualified unknown column (e.g. f.director) stops tracing. Fall back to a
        # conservative name check so a PII reference elsewhere still wins over "unknown".
        hit = _restricted_by_name(tree, restricted)
        if hit:
            return blocked(
                STAGE, "restricted_column", f"column {hit} is restricted (PII) and never exposed"
            )
        return repair(STAGE, "unknown_column", f"{_first_line(exc)} (not an allowlisted column)")
    except SqlglotError as exc:
        # The tree parsed and passed AST_CHECK, so failing to resolve it here is "unsure".
        return blocked(STAGE, "unresolvable", f"could not resolve query scopes: {exc}")

    base_tables = {
        source.name
        for scope in scopes
        for source in scope.sources.values()
        if isinstance(source, exp.Table)
    }
    for scope in scopes:
        for column in scope.columns:
            if not column.meta.get(_WRITTEN):
                continue
            hit = _restricted_target(scope, column, restricted, base_tables)
            if hit:
                return blocked(
                    STAGE,
                    "restricted_column",
                    f"column {hit} is restricted (PII) and never exposed",
                )
            if not column.table and column.name in scope.sources:
                return blocked(
                    STAGE,
                    "whole_row_reference",
                    f"whole-row reference {column.name!r} would read every column",
                )

    # qualify turns a bare table alias used as a value (`SELECT c FROM customer c`) into this.
    whole_row = traced.find(exp.TableColumn)
    if whole_row is not None:
        return blocked(
            STAGE,
            "whole_row_reference",
            f"whole-row reference {whole_row.name!r} would read every column",
        )

    for scope in scopes:
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and source.name not in allowlist.tables:
                return repair(
                    STAGE,
                    "unknown_table",
                    f"table {source.name!r} does not exist; allowed tables: "
                    + ", ".join(sorted(allowlist.tables)),
                )

    # Pass 2: validate against allowlisted columns only and expand * to them.
    expanded = tree.copy()
    try:
        expanded = _qualify(expanded, _schema(allowlist, with_restricted=False), validate=True)
    except OptimizeError as exc:
        return repair(STAGE, "unknown_column", f"{_first_line(exc)} (not an allowlisted column)")
    except SqlglotError as exc:
        return blocked(STAGE, "unresolvable", f"could not qualify query: {exc}")

    for star in expanded.find_all(exp.Star):
        if isinstance(star.parent, exp.Column):
            return blocked(
                STAGE,
                "whole_row_reference",
                f"{star.parent.sql(dialect=DIALECT)} would read every column",
            )
        if not isinstance(star.parent, exp.Count):
            return blocked(STAGE, "whole_row_reference", "* could not be expanded to columns")

    if _has_star(tree):
        return passed(STAGE, "star_expanded", "* expanded to allowlisted columns only", expanded)
    return passed(STAGE, "schema_ok", "all tables and columns are allowlisted", expanded)


def _qualify(tree: exp.Expr, schema: dict[str, object], *, validate: bool) -> exp.Expr:
    result: exp.Expr = qualify(
        tree,
        schema=schema,
        dialect=DIALECT,
        expand_stars=True,
        identify=False,
        validate_qualify_columns=validate,
    )
    return result


def _schema(allowlist: Allowlist, *, with_restricted: bool) -> dict[str, object]:
    schema: dict[str, object] = {}
    for name, table in allowlist.tables.items():
        columns = {col: "UNKNOWN" for col in table.columns}
        if with_restricted:
            columns |= {col: "UNKNOWN" for col in table.restricted}
        schema[name] = columns
    return schema


def _restricted_target(
    scope: Scope,
    column: exp.Column,
    restricted: dict[str, frozenset[str]],
    base_tables: set[str],
) -> str | None:
    """`table.column` if this column reads a restricted base column, else None."""
    if not column.table:
        # Unresolved (e.g. ambiguous between joined tables): fail closed on the name.
        for table in sorted(base_tables):
            if column.name in restricted.get(table, ()):
                return f"{table}.{column.name}"
        return None

    source: Any = scope.sources.get(column.table)
    if isinstance(source, exp.Table):
        return (
            f"{source.name}.{column.name}"
            if column.name in restricted.get(source.name, ())
            else None
        )
    if isinstance(source, Scope):
        # Derived table / CTE: follow the projection(s) that produce this column.
        branches = _branches(source)
        first = branches[0].expression
        names = [p.alias_or_name for p in first.selects] if isinstance(first, exp.Query) else []
        if column.name not in names:
            return None
        position = names.index(column.name)
        for branch in branches:
            selects = branch.expression.selects if isinstance(branch.expression, exp.Query) else []
            if position >= len(selects):
                continue
            for inner in selects[position].find_all(exp.Column):
                hit = _restricted_target(branch, inner, restricted, base_tables)
                if hit:
                    return hit
    return None


def _branches(scope: Scope) -> list[Scope]:
    """The SELECT scopes of a (possibly nested) UNION/INTERSECT/EXCEPT, else the scope itself."""
    if not scope.set_operation_scopes:
        return [scope]
    return [leaf for branch in scope.set_operation_scopes for leaf in _branches(branch)]


def _restricted_by_name(tree: exp.Expr, restricted: dict[str, frozenset[str]]) -> str | None:
    """Fallback: any column named like a restricted column of a table the query reads."""
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        for table in sorted(tables):
            if name in restricted.get(table, ()):
                return f"{table}.{name}"
    return None


def _has_star(tree: exp.Expr) -> bool:
    return any(not isinstance(s.parent, exp.Count) for s in tree.find_all(exp.Star))


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
