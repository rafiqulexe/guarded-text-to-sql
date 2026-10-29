"""Load prompt files from app/prompts/ (prompts are files, not inline strings).

A prompt file has a `## system` and a `## user` section. Placeholders use `$name`
(string.Template); a missing value is an error, never a silent blank.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Prompt:
    name: str
    system: Template
    user: Template

    def render(self, **values: str) -> tuple[str, str]:
        """(system, user) with every placeholder filled; KeyError if one is missing."""
        return self.system.substitute(values).strip(), self.user.substitute(values).strip()


@lru_cache
def load_prompt(name: str) -> Prompt:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    if set(sections) != {"system", "user"}:
        raise ValueError(f"prompt {name!r} needs exactly one '## system' and one '## user' section")
    return Prompt(name, Template(sections["system"]), Template(sections["user"]))


_TAG = re.compile(r"<(\s*/?\s*)(question|previous_sql|error|sql|rows|schema)\b", re.IGNORECASE)


def as_data(text: str) -> str:
    """Neutralise tag-like text in untrusted input so it cannot close or open our data tags."""
    return _TAG.sub(r"&lt;\1\2", text)
