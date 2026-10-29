"""Golden set: shape and coverage, and every expected SQL reproduces its recorded result
when run through all guards as the read-only role."""

import dataclasses
import re
from typing import Any

import pytest

from evals.golden import CATEGORIES, GoldenCase, check_case, load_cases, parse_cases

CASES = load_cases()
BANGLA = re.compile(r"[ঀ-৿]")


def test_coverage_minimums() -> None:
    english = [c for c in CASES if c.lang == "en"]
    per_category = {cat: sum(c.category == cat for c in english) for cat in CATEGORIES}
    assert len(english) >= 40
    assert per_category["easy"] >= 15
    assert per_category["joins"] >= 15
    assert per_category["aggregates_dates"] >= 10
    assert sum(c.lang == "bn" for c in CASES) >= 10


def test_lang_tag_matches_the_question_script() -> None:
    for case in CASES:
        assert bool(BANGLA.search(case.question)) == (case.lang == "bn"), case.id


def test_ids_and_questions_are_unique() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    assert len({(c.lang, c.question) for c in CASES}) == len(CASES)


def test_hashes_look_like_sha256() -> None:
    for case in CASES:
        assert re.fullmatch(r"[0-9a-f]{64}", case.result_hash), case.id


def _raw(**overrides: Any) -> dict[str, Any]:
    case = {
        "id": "x",
        "lang": "en",
        "category": "easy",
        "question": "How many films?",
        "sql": "SELECT count(*) FROM film",
        "row_count": 1,
        "result_hash": "0" * 64,
    }
    case.update(overrides)
    return {"cases": [case]}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_raw(lang="fr"), "lang"),
        (_raw(category="hard"), "category"),
        (_raw(row_count=0), "row_count"),
        (_raw(sql=" "), "sql"),
        (_raw(expected_sql="SELECT 1"), "unknown field"),
        (_raw(order_insensitive_cols="yes"), "order_insensitive_cols"),
        ({"cases": [_raw()["cases"][0], _raw()["cases"][0]]}, "duplicate"),
        ({"cases": "nope"}, "cases"),
    ],
)
def test_parser_rejects_malformed_cases(data: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_cases(data)


def test_parser_rejects_a_missing_field() -> None:
    data = _raw()
    del data["cases"][0]["result_hash"]
    with pytest.raises(ValueError, match="missing"):
        parse_cases(data)


# --- against the pinned Pagila data, as the read-only role ---------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_expected_sql_passes_guards_and_reproduces_the_result(
    case: GoldenCase, reader_url: str
) -> None:
    check = check_case(case, conninfo=reader_url)
    assert check.ok, check.detail


@pytest.mark.db
def test_a_wrong_expected_result_is_detected(reader_url: str) -> None:
    tampered = dataclasses.replace(CASES[0], result_hash="f" * 64)
    check = check_case(tampered, conninfo=reader_url)
    assert not check.ok and "result_hash" in check.detail
    recounted = dataclasses.replace(CASES[0], row_count=CASES[0].row_count + 1)
    assert not check_case(recounted, conninfo=reader_url).ok


@pytest.mark.db
def test_expected_sql_that_reads_a_restricted_column_is_rejected(reader_url: str) -> None:
    pii = dataclasses.replace(CASES[0], sql="SELECT email FROM customer")
    check = check_case(pii, conninfo=reader_url)
    assert not check.ok and "restricted_column" in check.detail
