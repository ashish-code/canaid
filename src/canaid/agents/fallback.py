"""Fallback agent.

Handles low-confidence intent, small talk, and any case not yet routed to a
specialist. Replaces the Phase 1 ``echo`` agent: the system prompt + the
streaming wrapper migrate here, plus we now inject the full conversation
history so multi-turn coherence works.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

from canaid.graph.state import State
from canaid.llm.lc import get_chat_model

SYSTEM_PROMPT = """You are a contact-center assistant for HealthSupplyCo, a B2B \
healthcare supply-chain distributor. You help prospective and existing clients \
with product information, account questions, and order inquiries.

Rules:
- Never give clinical or medical advice. Refer such questions to a clinician.
- Do not quote specific prices. Defer pricing to a sales contact.
- Be concise, professional, and warm. Aim for under 120 words unless asked.
- If unsure, say so and offer to connect the user with a human teammate.
"""


async def fallback_node(state: State) -> dict[str, Any]:
    chat = get_chat_model("supervisor", temperature=0.3, max_tokens=500)
    history = state.get("messages", [])
    response = await chat.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *history])
    return {"messages": [response]}
