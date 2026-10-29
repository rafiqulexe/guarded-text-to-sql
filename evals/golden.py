"""Golden set: questions with a verified expected answer (evals/golden.yaml).

`uv run python -m evals.golden` runs every case's expected SQL through all guards, executes the
guarded SQL as the read-only role and compares row count and result hash with the stored ones.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from app.pipeline.consistency import guard_sql, result_hash
from app.pipeline.execute import execute
from app.pipeline.result import Outcome

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.yaml"

Lang = Literal["en", "bn"]
Category = Literal["easy", "joins", "aggregates_dates"]
LANGS: tuple[Lang, ...] = ("en", "bn")
CATEGORIES: tuple[Category, ...] = ("easy", "joins", "aggregates_dates")
_FIELDS = {
    "id",
    "lang",
    "category",
    "question",
    "sql",
    "row_count",
    "result_hash",
    "order_insensitive_cols",
}


@dataclass(frozen=True)
class GoldenCase:
    id: str
    lang: Lang
    category: Category
    question: str
    sql: str
    row_count: int
    result_hash: str
    order_insensitive_cols: bool = False


@dataclass(frozen=True)
class CaseCheck:
    case: GoldenCase
    ok: bool
    detail: str


def parse_cases(data: Any) -> list[GoldenCase]:
    """Validate the YAML structure strictly: a typo in a field name must not pass silently."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("golden set must be a mapping with a `cases` list")
    cases: list[GoldenCase] = []
    for i, raw in enumerate(data["cases"]):
        if not isinstance(raw, dict):
            raise ValueError(f"case #{i} is not a mapping")
        where = f"case {raw.get('id', f'#{i}')}"
        if unknown := set(raw) - _FIELDS:
            raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
        if missing := (_FIELDS - {"order_insensitive_cols"}) - set(raw):
            raise ValueError(f"{where}: missing field(s) {sorted(missing)}")
        if raw["lang"] not in LANGS:
            raise ValueError(f"{where}: lang must be one of {LANGS}")
        if raw["category"] not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {CATEGORIES}")
        for name in ("id", "question", "sql", "result_hash"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError(f"{where}: {name} must be a non-empty string")
        if not isinstance(raw["row_count"], int) or raw["row_count"] < 1:
            raise ValueError(f"{where}: row_count must be a positive integer")
        insensitive = raw.get("order_insensitive_cols", False)
        if not isinstance(insensitive, bool):
            raise ValueError(f"{where}: order_insensitive_cols must be true or false")
        cases.append(
            GoldenCase(
                id=raw["id"],
                lang=raw["lang"],
                category=raw["category"],
                question=raw["question"],
                sql=raw["sql"],
                row_count=raw["row_count"],
                result_hash=raw["result_hash"],
                order_insensitive_cols=insensitive,
            )
        )
    ids = [c.id for c in cases]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate case id(s): {dupes}")
    return cases


def load_cases(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    with path.open(encoding="utf-8") as f:
        return parse_cases(yaml.safe_load(f))


def check_case(case: GoldenCase, *, conninfo: str | None = None) -> CaseCheck:
    """Expected SQL -> every guard -> guarded SQL executed read-only -> same rows as recorded."""
    guarded = guard_sql(case.sql, conninfo=conninfo)
    if guarded.outcome is not Outcome.PASSED:
        failure = guarded.failure
        rule = f"{failure.stage} {failure.rule}: {failure.detail}" if failure else "unknown"
        return CaseCheck(case, False, f"guards did not pass ({rule})")
    executed = execute(guarded.final_sql, conninfo=conninfo)
    if not executed.passed:
        return CaseCheck(case, False, f"execution failed ({executed.rule}: {executed.detail})")
    if len(executed.rows) != case.row_count:
        return CaseCheck(case, False, f"row_count {len(executed.rows)}, expected {case.row_count}")
    actual = result_hash(executed.rows, order_insensitive_cols=case.order_insensitive_cols)
    if actual != case.result_hash:
        return CaseCheck(
            case, False, f"result_hash {actual[:12]}…, expected {case.result_hash[:12]}…"
        )
    return CaseCheck(case, True, f"{len(executed.rows)} row(s)")


def main() -> int:
    cases = load_cases()
    checks = [check_case(case) for case in cases]
    for check in checks:
        mark = "ok  " if check.ok else "FAIL"
        print(
            f"{mark} {check.case.id:<12} {check.case.lang} {check.case.category:<16} {check.detail}"
        )
    failed = sum(not c.ok for c in checks)
    print(f"{len(checks) - failed}/{len(checks)} golden cases reproduce their expected result")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
