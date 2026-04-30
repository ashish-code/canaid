"""Stub-style nodes for routes without a full specialist agent.

After Phase 5, only `escalation` and `refusal` live here:
  * `escalation` is a real "summarize the convo and offer handoff" node.
  * `refusal` uses templated responses (`canaid.guardrails.refusals`)
    when ``state.refusal_reason`` is set, and falls back to an
    LLM-generated refusal when it isn't.

The honest-stub pattern was helpful in Phases 2-4 to keep the full graph
wired before all specialists existed; RAG (Phase 3) and Lookup (Phase 4)
have since been promoted to real agents.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from canaid.graph.state import State
from canaid.guardrails.refusals import refusal_text
from canaid.llm.lc import get_chat_model

_BASE = "You are HealthSupplyCo's contact-center assistant. "


_STUB_PROMPTS = {
    "escalation_node": _BASE
    + (
        "The user asked for a human or signaled frustration. Confirm warmly "
        "that you'll connect them, ask for the best email and a one-line "
        "summary of what they need help with so the teammate is briefed. "
        "Under 70 words."
    ),
    "refusal_node": _BASE
    + (
        "The user has asked for clinical/medical advice or something off-policy. "
        "Decline briefly and warmly, explain we don't provide clinical guidance, "
        "and offer the most useful alternative (a clinician for medical "
        "questions; a teammate for anything else). Under 60 words."
    ),
}


async def _stub(state: State, key: str) -> dict[str, Any]:
    chat = get_chat_model("summarizer", temperature=0.3, max_tokens=240)
    history = state.get("messages", [])
    response = await chat.ainvoke(
        [SystemMessage(content=_STUB_PROMPTS[key]), *history]
    )
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    return {"messages": [response]}


async def escalation_node(state: State) -> dict[str, Any]:
    return await _stub(state, "escalation_node")


async def refusal_node(state: State) -> dict[str, Any]:
    """Use a templated refusal when we know the reason; LLM fallback otherwise.

    The template path is *deterministic* — same input → same output, no
    network call, traceable in the audit log. We prefer it because refusal
    voice should be consistent across the product.
    """
    reason = state.get("refusal_reason")
    if reason:
        return {"messages": [AIMessage(content=refusal_text(reason))]}
    # No specific reason recorded yet — fall through to the LLM stub.
    return await _stub(state, "refusal_node")
