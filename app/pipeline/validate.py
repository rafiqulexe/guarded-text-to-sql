"""Judge LLM validation (VALIDATING): does the SQL + a preview of its rows answer the question?"""

from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    verdict: Literal["yes", "partial", "no"] = Field(
        description="does the SQL and its result answer the question?"
    )
    expects_rows: bool = Field(description="would a correct answer return at least one row?")
    reason: str = Field(description="one or two sentences explaining the verdict")
