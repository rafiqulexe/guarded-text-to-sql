"""Adversarial set: attacks that must never execute (evals/adversarial.yaml).

`uv run python -m evals.adversarial` runs every case's `attack_sql` (what a model that obeys the
attacker would write) through all guards and checks it is BLOCKED by the expected rule, without
relying on the DB permission layer.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from app.pipeline.consistency import guard_sql
from app.pipeline.result import Outcome, Severity

ADVERSARIAL_PATH = Path(__file__).resolve().parent / "adversarial.yaml"

Lang = Literal["en", "bn"]
Category = Literal[
    "destructive_sql",
    "stacked_statements",
    "resource_abuse",
    "data_exfiltration",
    "prompt_injection_question",
    "prompt_injection_data",
    "comment_smuggling",
    "pii_request",
]
Expected = Literal["BLOCKED", "UNANSWERABLE"]
LANGS: tuple[Lang, ...] = ("en", "bn")
CATEGORIES: tuple[Category, ...] = (
    "destructive_sql",
    "stacked_statements",
    "resource_abuse",
    "data_exfiltration",
    "prompt_injection_question",
    "prompt_injection_data",
    "comment_smuggling",
    "pii_request",
)
EXPECTED: tuple[Expected, ...] = ("BLOCKED", "UNANSWERABLE")
_REQUIRED = {"id", "lang", "category", "question", "expected"}
_OPTIONAL = {"attack_sql", "blocked_rule"}


@dataclass(frozen=True)
class AdversarialCase:
    id: str
    lang: Lang
    category: Category
    question: str
    expected: Expected
    attack_sql: str | None = None
    blocked_rule: str | None = None


@dataclass(frozen=True)
class AttackCheck:
    case: AdversarialCase
    ok: bool
    detail: str


def parse_cases(data: Any) -> list[AdversarialCase]:
    """Validate the YAML structure strictly: a typo in a field name must not pass silently."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("adversarial set must be a mapping with a `cases` list")
    cases: list[AdversarialCase] = []
    for i, raw in enumerate(data["cases"]):
        if not isinstance(raw, dict):
            raise ValueError(f"case #{i} is not a mapping")
        where = f"case {raw.get('id', f'#{i}')}"
        if unknown := set(raw) - _REQUIRED - _OPTIONAL:
            raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
        if missing := _REQUIRED - set(raw):
            raise ValueError(f"{where}: missing field(s) {sorted(missing)}")
        if raw["lang"] not in LANGS:
            raise ValueError(f"{where}: lang must be one of {LANGS}")
        if raw["category"] not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {CATEGORIES}")
        if raw["expected"] not in EXPECTED:
            raise ValueError(f"{where}: expected must be one of {EXPECTED}")
        for name in ("id", "question"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError(f"{where}: {name} must be a non-empty string")
        for name in _OPTIONAL & set(raw):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError(f"{where}: {name} must be a non-empty string")
        if ("attack_sql" in raw) != ("blocked_rule" in raw):
            raise ValueError(f"{where}: attack_sql and blocked_rule must be given together")
        cases.append(
            AdversarialCase(
                id=raw["id"],
                lang=raw["lang"],
                category=raw["category"],
                question=raw["question"],
                expected=raw["expected"],
                attack_sql=raw.get("attack_sql"),
                blocked_rule=raw.get("blocked_rule"),
            )
        )
    ids = [c.id for c in cases]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate case id(s): {dupes}")
    return cases


def load_cases(path: Path = ADVERSARIAL_PATH) -> list[AdversarialCase]:
    with path.open(encoding="utf-8") as f:
        return parse_cases(yaml.safe_load(f))


def check_attack(case: AdversarialCase, *, conninfo: str | None = None) -> AttackCheck:
    """attack_sql -> every guard -> BLOCKED by `blocked_rule`, never by DB permissions alone."""
    if case.attack_sql is None:
        return AttackCheck(case, True, "no SQL form")
    guarded = guard_sql(case.attack_sql, conninfo=conninfo)
    failure = guarded.failure
    if guarded.outcome is not Outcome.BLOCKED or failure is None:
        return AttackCheck(case, False, f"not blocked (outcome {guarded.outcome})")
    if failure.severity is Severity.CRITICAL:
        return AttackCheck(case, False, f"layer leak: only stopped by {failure.rule}")
    if failure.rule != case.blocked_rule:
        return AttackCheck(case, False, f"blocked by {failure.rule}, expected {case.blocked_rule}")
    return AttackCheck(case, True, f"BLOCKED at {failure.stage} ({failure.rule})")


def main() -> int:
    checks = [check_attack(case) for case in load_cases()]
    for check in checks:
        mark = "ok  " if check.ok else "FAIL"
        case = check.case
        print(f"{mark} {case.id:<18} {case.lang} {case.category:<26} {check.detail}")
    failed = sum(not c.ok for c in checks)
    print(f"{len(checks) - failed}/{len(checks)} attacks are stopped by the guards")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
