"""Per-model pricing table and cost calculation.

We track cost as a first-class metric for two reasons:
  1. Different agents use different models (per-agent registry); the
     per-turn cost mix tells us whether the routing is sending traffic
     to the right seat.
  2. The Bedrock prompt-cache savings show up as reduced input-token
     billing — without a cost meter you can't tell whether prompt caching
     is actually firing.

Prices are USD per 1M tokens, current as of mid-2025 for cross-region
inference profiles in us-east-1. Override via env if your contract
differs (or if you're seeing this in a future where rates have moved).

If a model isn't in the table we return ``0.0`` (and warn) — better to
under-report than to report fictional numbers.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import structlog

log = structlog.get_logger(__name__)


class _Rate(NamedTuple):
    input_per_mtok: float
    output_per_mtok: float


# Keyed by normalized model family — see _normalize_model_key.
_PRICING: dict[str, _Rate] = {
    "anthropic.claude-sonnet-4-5": _Rate(3.00, 15.00),
    "anthropic.claude-sonnet-4-6": _Rate(3.00, 15.00),
    "anthropic.claude-haiku-4-5": _Rate(1.00, 5.00),
    "anthropic.claude-opus-4-1": _Rate(15.00, 75.00),
    "anthropic.claude-opus-4-7": _Rate(15.00, 75.00),
    "meta.llama3-3-70b-instruct": _Rate(0.72, 0.72),
    "amazon.nova-lite": _Rate(0.06, 0.24),
    "amazon.nova-pro": _Rate(0.80, 3.20),
    "amazon.titan-embed-text": _Rate(0.02, 0.00),
}


def _normalize_model_key(model_id: str) -> str:
    """Strip cross-region prefix and date/version suffix.

    Examples:
      ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` → ``anthropic.claude-sonnet-4-5``
      ``meta.llama3-3-70b-instruct-v1:0``              → ``meta.llama3-3-70b-instruct``
      ``amazon.titan-embed-text-v2:0``                 → ``amazon.titan-embed-text-v2``
    """
    parts = model_id.split(".")
    if parts and parts[0] in {"us", "eu", "apac"}:
        parts = parts[1:]
    if not parts:
        return ""
    last = parts[-1]
    # Drop trailing date (-YYYYMMDD…) or version (-vN[:M]) suffixes.
    last = re.sub(r"-\d{8}.*$", "", last)
    last = re.sub(r"-v\d+(:.*)?$", "", last)
    parts[-1] = last
    return ".".join(parts)


def cost_for(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Returns the USD cost for a single LLM call. Always non-negative."""
    if not model_id or input_tokens < 0 or output_tokens < 0:
        return 0.0
    key = _normalize_model_key(model_id)
    rate = _PRICING.get(key)
    if rate is None:
        log.warning("cost.unknown_model", model_id=model_id, normalized=key)
        return 0.0
    return (input_tokens / 1_000_000.0) * rate.input_per_mtok + (
        output_tokens / 1_000_000.0
    ) * rate.output_per_mtok


def known_models() -> list[str]:
    return sorted(_PRICING.keys())
