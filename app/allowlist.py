"""Typed view of db/allowlist.yaml, the single source for tables, columns and functions."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "db" / "allowlist.yaml"


@dataclass(frozen=True)
class TableSpec:
    name: str
    description: str
    columns: dict[str, dict[str, Any]]
    restricted: frozenset[str]
    foreign_keys: dict[str, str]


@dataclass(frozen=True)
class Allowlist:
    tables: dict[str, TableSpec]
    functions: frozenset[str]

    def restricted_columns(self) -> dict[str, frozenset[str]]:
        return {name: t.restricted for name, t in self.tables.items() if t.restricted}


def parse_allowlist(data: dict[str, Any]) -> Allowlist:
    tables = {
        name: TableSpec(
            name=name,
            description=spec.get("description", ""),
            columns=dict(spec["columns"]),
            restricted=frozenset(spec.get("restricted", [])),
            foreign_keys=dict(spec.get("foreign_keys", {})),
        )
        for name, spec in data["tables"].items()
    }
    return Allowlist(tables=tables, functions=frozenset(f.lower() for f in data["functions"]))


@lru_cache
def load_allowlist(path: Path = DEFAULT_PATH) -> Allowlist:
    return parse_allowlist(yaml.safe_load(path.read_text(encoding="utf-8")))
