"""Builds the compact schema prompt from the allowlist.

Only allowlisted tables and columns appear. Restricted (PII) columns are left out entirely,
not even named; a neutral note tells the model such data is not available, so it can answer
`answerable: false` instead of guessing.
"""

from app.allowlist import Allowlist, TableSpec

HEADER = (
    "PostgreSQL database (Pagila, a DVD rental store). These are the ONLY tables and columns "
    "you may use.\n"
    "Format: table — description, then one column per item as `name type`; PK = primary key; "
    "`-> table.column` = join key; `in (...)` = the only allowed values.\n"
)
FOOTER = (
    "Not available in this database: personal contact details (e-mail addresses, telephone "
    "numbers), staff login data, and anything not listed above."
)

# Conservative chars-per-token for the offline estimate (real tokenizers average ~3.5-4 on
# this kind of text); the exact count comes from the provider in scripts/live_llm_check.py.
CHARS_PER_TOKEN = 3.0


def build_schema_context(allowlist: Allowlist) -> str:
    blocks = [_table(table) for table in allowlist.tables.values()]
    return HEADER + "\n" + "\n".join(blocks) + "\n" + FOOTER + "\n"


def _table(table: TableSpec) -> str:
    columns = [
        _column(name, spec, table.foreign_keys.get(name)) for name, spec in table.columns.items()
    ]
    return f"{table.name} — {table.description}\n  " + "; ".join(columns) + "\n"


def _column(name: str, spec: dict[str, object], foreign_key: str | None) -> str:
    parts = [name, str(spec.get("type", "?"))]
    if spec.get("pk"):
        parts.append("PK")
    if foreign_key:
        parts.append(f"-> {foreign_key}")
    values = spec.get("values")
    if isinstance(values, list):
        parts.append("in (" + ", ".join(str(v) for v in values) + ")")
    description = spec.get("description")
    if description:
        parts.append(f"({description})")
    return " ".join(parts)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1
