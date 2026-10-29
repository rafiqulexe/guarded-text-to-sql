"""Guard outcomes, named after the run states."""

from dataclasses import dataclass
from enum import StrEnum

from sqlglot import exp


class Stage(StrEnum):
    INPUT_CHECK = "INPUT_CHECK"
    AST_CHECK = "AST_CHECK"
    SCHEMA_CHECK = "SCHEMA_CHECK"
    REWRITING = "REWRITING"
    DRY_RUN = "DRY_RUN"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"


class Outcome(StrEnum):
    PASSED = "passed"  # continue to the next state
    REPAIR = "repair"  # back to GENERATING with the error text (counts toward MAX_REPAIRS)
    BLOCKED = "BLOCKED"  # terminal safety stop, never retried
    FAILED = "FAILED"  # terminal, unrecoverable (e.g. timeout)


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"
    CRITICAL = "critical"  # a DB permission caught what the guards missed


@dataclass(frozen=True)
class GuardResult:
    stage: Stage
    outcome: Outcome
    rule: str
    detail: str
    severity: Severity
    expression: exp.Expr | None = None

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASSED


def passed(stage: Stage, rule: str, detail: str, expression: exp.Expr | None = None) -> GuardResult:
    return GuardResult(stage, Outcome.PASSED, rule, detail, Severity.INFO, expression)


def repair(stage: Stage, rule: str, detail: str) -> GuardResult:
    return GuardResult(stage, Outcome.REPAIR, rule, detail, Severity.WARN)


def blocked(stage: Stage, rule: str, detail: str, *, critical: bool = False) -> GuardResult:
    severity = Severity.CRITICAL if critical else Severity.BLOCK
    return GuardResult(stage, Outcome.BLOCKED, rule, detail, severity)


def failed(stage: Stage, rule: str, detail: str) -> GuardResult:
    return GuardResult(stage, Outcome.FAILED, rule, detail, Severity.WARN)
