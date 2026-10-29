"""Compact schema prompt built from allowlist.yaml (PLAN 3.1)."""

import re

from app.allowlist import load_allowlist, parse_allowlist
from app.schema_context import build_schema_context, estimate_tokens

ALLOWLIST = load_allowlist()
CONTEXT = build_schema_context(ALLOWLIST)


def _table_block(name: str) -> str:
    match = re.search(rf"^{name} — .*?(?=^\S|\Z)", CONTEXT, re.MULTILINE | re.DOTALL)
    assert match, f"no block for {name}"
    return match.group(0)


def test_every_allowlisted_table_and_column_is_listed() -> None:
    for name, table in ALLOWLIST.tables.items():
        block = _table_block(name)
        for column in table.columns:
            assert re.search(rf"\b{column}\b", block), f"{name}.{column} missing"


def test_restricted_columns_are_never_mentioned() -> None:
    restricted = {c for t in ALLOWLIST.tables.values() for c in t.restricted}
    assert restricted  # email, phone, password, username, picture
    for column in restricted:
        assert not re.search(rf"\b{column}\b", CONTEXT, re.IGNORECASE), column


def test_types_keys_foreign_keys_and_enum_values() -> None:
    film = _table_block("film")
    assert "film_id int PK" in film
    assert "language_id int -> language.language_id" in film
    assert "G, PG, PG-13, R, NC-17" in film
    assert "Trailers" in film and "text[]" in film
    assert "PK" in _table_block("film_actor")  # composite key columns marked too


def test_descriptions_and_payment_note_are_included() -> None:
    assert "Movies available for rent" in CONTEXT
    assert "never its partitions" in _table_block("payment")
    assert "rental period in days" in _table_block("film")


def test_context_says_what_is_not_available() -> None:
    assert "not available" in CONTEXT.lower()


def test_tables_follow_allowlist_order_and_output_is_deterministic() -> None:
    order = [m.group(1) for m in re.finditer(r"^(\w+) — ", CONTEXT, re.MULTILINE)]
    assert order == list(ALLOWLIST.tables)
    assert build_schema_context(ALLOWLIST) == CONTEXT


def test_built_from_the_allowlist_not_hardcoded() -> None:
    tiny = parse_allowlist(
        {
            "tables": {
                "actor": {
                    "description": "People",
                    "columns": {"actor_id": {"type": "int", "pk": True}},
                }
            },
            "functions": ["count"],
        }
    )
    text = build_schema_context(tiny)
    assert "actor — People" in text and "film" not in text


def test_acceptance_prompt_context_under_2500_tokens() -> None:
    """PLAN 3.1: conservative offline estimate; exact count: scripts/live_llm_check.py."""
    assert estimate_tokens(CONTEXT) < 2500
