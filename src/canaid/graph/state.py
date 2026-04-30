"""LangGraph state schema.

Single dict that flows through every node. We grow it phase by phase rather
than redesigning per-feature — each new field gets a sensible default so
adding a node never requires touching unrelated code.

Two reducers do the work:
  * ``messages``      — uses LangGraph's ``add_messages`` so node returns
                         like ``{"messages": [AIMessage(...)]}`` *append*
                         instead of replacing the list.
  * everything else   — last-write-wins. A node that wants to update
                         ``intent`` just returns ``{"intent": "..."}``.

Phase fan-in:
  Phase 2 — messages, user_type, intent, confidence, route, refusal_reason
  Phase 3 — citations
  Phase 4 — tool_calls, tool_results, account_id
  Phase 5 — pii_redactions, refusal_reason (extended)
  Phase 6 — checkpointing happens via the LangGraph checkpointer, not state
  Phase 8 — cost_so_far, latency_so_far, span_id
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

UserType = Literal["new", "existing", "unknown"]
Intent = Literal[
    "lead_qualification",   # new prospect — drive BANT
    "catalog_question",     # product / SKU / policy lookup → RAG
    "account_lookup",       # existing client — order / account info → tool-use
    "escalation",           # explicit human handoff request
    "small_talk",           # greetings, thanks, off-topic but harmless
    "refusal",              # clinical advice / off-policy
    "unknown",              # could not classify confidently
]
Route = Literal["qualifier", "rag", "lookup", "escalation", "fallback", "refusal"]


class State(TypedDict, total=False):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Classification (set by intent node)
    user_type: UserType
    intent: Intent
    confidence: float
    rationale: str

    # Routing decision (set by supervisor)
    route: Route

    # Refusal context (Phase 5 will populate)
    refusal_reason: str | None

    # Phase 3 placeholders
    citations: list[dict[str, Any]]

    # Phase 4 placeholders
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    account_id: str | None

    # Phase 8 placeholders — running totals across the turn
    cost_usd: float
    latency_ms: int
