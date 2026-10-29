"""AST_CHECK (layer 1): sqlglot rules.

Fail closed: anything that is not provably a single read-only SELECT is BLOCKED. Only a real
parse error goes back to GENERATING. Later stages must execute SQL regenerated
from the returned tree, never the original text, so lexer tricks (nested comments, dollar
quotes) can't make the DB see something different from what was checked.
"""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.allowlist import Allowlist, load_allowlist
from app.pipeline.result import GuardResult, Stage, blocked, passed, repair

DIALECT = "postgres"
STAGE = Stage.AST_CHECK

# SQL syntax that sqlglot models as function nodes but that is not a function call. Exact
# classes only, never a base class: many sqlglot "operator" classes are real function calls
# (corr(), array_position(), POWER()). Their arguments are still checked. Every other
# function node must be on the allowlist.
SYNTAX_NODES: tuple[type[exp.Expr], ...] = (
    exp.Case,  # CASE ... END
    exp.If,  # a WHEN ... THEN branch, exempt only directly inside CASE
    exp.Array,  # ARRAY[...] / ARRAY(subquery)
    exp.Exists,  # EXISTS (subquery)
    exp.And,  # AND
    exp.Or,  # OR
    exp.ArrayContainsAll,  # @>   (special_features text[])
    exp.ArrayContainedBy,  # <@
    exp.ArrayOverlaps,  # &&
)

# Statement types that may never appear anywhere in the tree.
FORBIDDEN_STATEMENTS: tuple[type[exp.Expr], ...] = (
    exp.DML,
    exp.DDL,
    exp.Command,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Use,
    exp.Pragma,
    exp.Describe,
    exp.Show,
    exp.Kill,
    exp.LoadData,
    exp.Cache,
    exp.Uncache,
    exp.Refresh,
    exp.Analyze,
    exp.Execute,
    exp.Declare,
)

# Leading function name of a rendered call: `DATE_TRUNC(...)`, `CURRENT_DATE`.
_CALL_NAME = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\(|$)")


def check(sql: str, allowlist: Allowlist | None = None) -> GuardResult:
    allowlist = allowlist or load_allowlist()
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except SqlglotError as exc:
        return repair(STAGE, "parse_error", f"could not parse SQL: {_first_line(exc)}")
    if not statements:
        return repair(STAGE, "empty_sql", "no SQL statement found")
    if len(statements) > 1:
        return blocked(
            STAGE,
            "single_statement",
            f"{len(statements)} statements found; exactly one SELECT is allowed",
        )

    tree = statements[0]
    for rule_check in (_check_shape, _check_functions, _check_relations):
        result = rule_check(tree, allowlist)
        if result is not None:
            return result
    return passed(STAGE, "ast_ok", "single read-only SELECT", tree)


def _check_shape(tree: exp.Expr, allowlist: Allowlist) -> GuardResult | None:
    root = tree
    while isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        return blocked(
            STAGE, "select_only", f"{type(root).__name__} statement is not allowed; only SELECT"
        )

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_STATEMENTS):
            if node.find_ancestor(exp.CTE) is not None:
                return blocked(
                    STAGE,
                    "data_modifying_cte",
                    f"{type(node).__name__} inside a WITH clause is not allowed",
                )
            return blocked(STAGE, "select_only", f"{type(node).__name__} statement is not allowed")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            return blocked(STAGE, "recursive_cte", "WITH RECURSIVE is not allowed")
        if isinstance(node, exp.CTE) and not isinstance(node.this, exp.Query):
            return blocked(STAGE, "data_modifying_cte", "a WITH clause may only contain a SELECT")
        if isinstance(node, exp.Subquery) and not isinstance(node.this, exp.Query):
            return blocked(STAGE, "select_only", "a subquery may only contain a SELECT")
        if isinstance(node, exp.Into):
            return blocked(STAGE, "select_into", "SELECT ... INTO creates a table; not allowed")
        if isinstance(node, exp.Lock):
            return blocked(STAGE, "locking_clause", "FOR UPDATE/SHARE row locks are not allowed")
    return None


def _check_functions(tree: exp.Expr, allowlist: Allowlist) -> GuardResult | None:
    for node in tree.find_all(exp.Func):
        if _is_syntax(node):
            continue
        if isinstance(node.parent, exp.Dot) and node.parent.expression is node:
            return blocked(
                STAGE,
                "function_not_allowed",
                f"schema-qualified function call {node.parent.sql(dialect=DIALECT)} is not allowed",
            )
        # Check the name as it will be executed (the tree is re-rendered before execution), so
        # aliases resolve to what really runs: date_part() -> EXTRACT, now() -> CURRENT_TIMESTAMP.
        name = _rendered_name(node)
        if name is None or name not in allowlist.functions:
            shown = name or type(node).__name__
            return blocked(
                STAGE, "function_not_allowed", f"function {shown}() is not in the allowlist"
            )
    return None


def _check_relations(tree: exp.Expr, allowlist: Allowlist) -> GuardResult | None:
    for node in tree.find_all(exp.Table, exp.Column):
        catalog, schema = node.args.get("catalog"), node.args.get("db")
        if catalog is not None:
            return blocked(
                STAGE,
                "system_catalog",
                f"cross-database reference {node.sql(dialect=DIALECT)} is not allowed",
            )
        if schema is not None and _normalized(schema) != "public":
            return blocked(
                STAGE,
                "system_catalog",
                f"schema {schema.sql(dialect=DIALECT)} is not allowed; only public tables",
            )
        if isinstance(node, exp.Table) and node.name.lower().startswith("pg_"):
            return blocked(STAGE, "system_catalog", f"system catalog {node.name} is not allowed")
    return None


def _normalized(identifier: exp.Expr) -> str:
    """Postgres identifier rules: unquoted folds to lower case, quoted is exact."""
    if isinstance(identifier, exp.Identifier) and identifier.quoted:
        return identifier.name
    return identifier.name.lower()


def _is_syntax(node: exp.Func) -> bool:
    if type(node) not in SYNTAX_NODES:  # exact type: subclasses are not exempt
        return False
    # IF(a, b, c) on its own is a function call; it is only syntax as a CASE branch.
    return not isinstance(node, exp.If) or isinstance(node.parent, exp.Case)


def _rendered_name(node: exp.Func) -> str | None:
    """Function name in the Postgres SQL this node renders to, or None if it doesn't render
    as a plain call (fail closed: then it is not allowed)."""
    match = _CALL_NAME.match(node.sql(dialect=DIALECT, comments=False))
    return match.group(1).lower() if match else None


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
