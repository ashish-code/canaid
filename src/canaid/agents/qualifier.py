"""Lead Qualifier agent.

Stub for Phase 2: a streaming response node that runs BANT
(Budget / Authority / Need / Timeline) discovery for new prospects. The
real qualifier (Phase 7+) will track which BANT slots are filled in the
graph state and skip past ones the user has already answered. For now we
simply stream a context-aware reply and let the supervisor model handle
multi-turn coherence.

Implementation uses ``ChatBedrockConverse`` so token streaming flows
through LangGraph's ``astream_events`` and on out as SSE.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

from canaid.graph.state import State
from canaid.llm.lc import get_chat_model

SYSTEM_PROMPT = """You are HealthSupplyCo's lead qualifier in the contact-center \
chatbot. The person you're speaking with is a NEW prospective client (a \
healthcare facility — hospital, clinic, pharmacy, long-term-care home).

Your goal in this short turn: move the conversation toward BANT discovery — \
Budget, Authority, Need, Timeline — without sounding like a sales script. Ask \
ONE focused question per turn. Reflect back what they've shared.

Hard rules:
- Never give clinical or medical advice.
- Do not quote prices. If asked, say a sales contact can prepare a quote.
- Be concise and warm. Aim for under 90 words.
- If they want to talk to a person, offer to escalate immediately.
- If you don't know something, say so.

Open with a brief acknowledgement, then your one question. If they've already \
shared some BANT info in earlier turns, don't ask the same thing again.
"""


async def qualifier_node(state: State) -> dict[str, Any]:
    """Stream a qualifier reply. Returns the appended AIMessage via add_messages."""
    chat = get_chat_model("qualifier", temperature=0.4, max_tokens=400)
    history = state.get("messages", [])
    response = await chat.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *history])
    return {"messages": [response]}
