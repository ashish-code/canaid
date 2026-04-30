"""Per-agent model registry.

Maps a logical agent name (`supervisor`, `intent`, `rag`, …) to a
`ModelSpec` — model id, vendor, capability flags, cost tier. Centralizing
the mapping means:

  1. Per-agent model swaps happen via env vars, not code edits.
  2. Routing / fallback / cost-budgeting code can interrogate the spec
     instead of pattern-matching strings.
  3. The "different LLMs per agent" claim is auditable — `get_all_specs()`
     lists exactly which model each agent uses.

Vendor diversity is intentional. Anthropic models for reasoning paths,
Meta for tool use, Amazon Nova for cheap utility paths. If you swap them
to all-Sonnet you'll still work; you just lose the cross-vendor
demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from canaid.config import Settings, get_settings

AgentName = Literal[
    "supervisor", "intent", "qualifier", "rag", "tool", "summarizer"
]

CostTier = Literal["low", "mid", "high"]


@dataclass(frozen=True)
class ModelSpec:
    agent: AgentName
    model_id: str
    vendor: str           # anthropic / meta / amazon / mistral / cohere / ai21
    supports_tools: bool  # whether the model supports Converse toolConfig
    cost_tier: CostTier


def _vendor_of(model_id: str) -> str:
    parts = model_id.split(".")
    # Cross-region inference profile prefixes: us. / eu. / apac.
    if parts and parts[0] in {"us", "eu", "apac"}:
        parts = parts[1:]
    head = parts[0] if parts else ""
    for v in ("anthropic", "meta", "amazon", "mistral", "cohere", "ai21"):
        if head.startswith(v):
            return v
    return "unknown"


# Capability + cost defaults per agent. Override `model_id` in env;
# `supports_tools` and `cost_tier` are intrinsic to the agent's role,
# not the model — i.e., we use them to encode "what we expect from
# this seat", not "what this exact model can do".
_DEFAULTS: dict[AgentName, tuple[bool, CostTier]] = {
    "supervisor": (True, "high"),
    "intent": (False, "low"),
    "qualifier": (True, "high"),
    "rag": (True, "high"),
    "tool": (True, "mid"),
    "summarizer": (False, "low"),
}


def get_model_spec(agent: AgentName, settings: Settings | None = None) -> ModelSpec:
    s = settings or get_settings()
    model_id_by_agent: dict[AgentName, str] = {
        "supervisor": s.supervisor_model,
        "intent": s.intent_model,
        "qualifier": s.qualifier_model,
        "rag": s.rag_model,
        "tool": s.tool_model,
        "summarizer": s.summarizer_model,
    }
    model_id = model_id_by_agent[agent]
    supports_tools, tier = _DEFAULTS[agent]
    return ModelSpec(
        agent=agent,
        model_id=model_id,
        vendor=_vendor_of(model_id),
        supports_tools=supports_tools,
        cost_tier=tier,
    )


def get_all_specs(settings: Settings | None = None) -> list[ModelSpec]:
    return [get_model_spec(a, settings) for a in _DEFAULTS]
