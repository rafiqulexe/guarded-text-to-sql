"""Token and cost accounting per LLM call and per run."""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from app.config import MODEL_PRICES_USD_PER_MTOK
from app.pipeline.llm import LLMError, LLMResponse

Purpose = Literal["generate", "repair", "judge"]
_MICRO = Decimal("0.000001")  # numeric(10, 6)


def list_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Cost at list prices, or None when the model is not in the price table."""
    price = MODEL_PRICES_USD_PER_MTOK.get(model)
    if price is None:
        return None
    usd = (
        Decimal(str(price[0])) * input_tokens + Decimal(str(price[1])) * output_tokens
    ) / 1_000_000
    return usd.quantize(_MICRO, rounding=ROUND_HALF_UP)


@dataclass
class Ledger:
    """All LLM calls of one run; `fields()` gives the query_runs columns to write."""

    billing: Literal["free", "paid"]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        purpose: Purpose,
        requested_model: str,
        *,
        candidate: int | None = None,
        response: LLMResponse[Any] | None = None,
        error: LLMError | None = None,
    ) -> None:
        if response is not None:
            cost = list_cost(response.model, response.input_tokens, response.output_tokens)
            entry = {
                "purpose": purpose,
                "candidate": candidate,
                "requested_model": requested_model,
                "model": response.model,  # the model that actually answered
                "fallback_from": response.fallback_from,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "attempts": response.attempts,
                "cost_usd_list": None if cost is None else str(cost),
                "price_known": cost is not None,
                "error": None,
            }
        else:
            entry = {
                "purpose": purpose,
                "candidate": candidate,
                "requested_model": requested_model,
                "model": None,
                "fallback_from": None,
                "input_tokens": 0,  # a failed call reports no usage
                "output_tokens": 0,
                "attempts": None,
                "cost_usd_list": "0",
                "price_known": True,
                "error": f"{error.kind}: {error}" if error is not None else "unknown error",
            }
        self.calls.append(entry)

    @property
    def input_tokens(self) -> int:
        return sum(int(c["input_tokens"]) for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(int(c["output_tokens"]) for c in self.calls)

    @property
    def cost_usd_list(self) -> Decimal:
        return sum((Decimal(c["cost_usd_list"] or "0") for c in self.calls), Decimal(0))

    @property
    def cost_usd(self) -> Decimal:
        return self.cost_usd_list if self.billing == "paid" else Decimal(0)

    def fields(self) -> dict[str, Any]:
        return {
            "llm_calls": list(self.calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "cost_usd_list": self.cost_usd_list,
        }
