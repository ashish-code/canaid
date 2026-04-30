"""Account Lookup agent (tool-using).

Walks a tool-use loop against the Llama 3.3 70B model on Bedrock:

  1. Send conversation + tool specs.
  2. If the model returns tool_calls, invoke each one, append the
     ``ToolMessage`` result, send the conversation back.
  3. Repeat until the model returns a plain assistant message (or we hit
     the safety cap).

We chose Llama for this seat for two reasons that matter to the showcase:
  * Cross-vendor portability — proof we're not Anthropic-locked.
  * Tool-call accuracy on Llama 3.3 is competitive with Anthropic at
    a lower per-call cost (good for a routinely-fired specialist).

LangChain's ``@tool``-decorated functions fire ``on_tool_start`` /
``on_tool_end`` callbacks during ``astream_events``. The API server picks
those up and forwards them as ``tool_call`` / ``tool_result`` SSE frames
so the UI can render the tool trace inline.

Production hardening (Phase 5+):
  * The agent should never call a tool with raw user-supplied PII unless
    Comprehend has classified it. Today the tool args come from the LLM
    based on user prose, so this is mostly bounded — but we'll add a
    guardrail step before tool invocation in Phase 5.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from canaid.graph.state import State
from canaid.llm.lc import get_chat_model
from canaid.tools.lookup_tools import LOOKUP_TOOLS, TOOL_MAP

log = structlog.get_logger(__name__)

_MAX_TOOL_ITERATIONS = 4

SYSTEM_PROMPT = """You are HealthSupplyCo's account-lookup specialist in the \
contact-center chatbot. You help an EXISTING client check their account \
status, recent orders, and order status.

You have three tools:
- lookup_account(account_id|email|facility_name)
- get_recent_orders(account_id, limit?)
- get_order_status(order_id)

Workflow:
1. If the user has not identified themselves, ask for ONE piece of identifying \
information first (account ID, registered email, or facility name). Don't ask \
multiple verification questions at once.
2. Call ``lookup_account`` with what they gave you.
3. If the user asks about orders, call the relevant order tool with the \
   account_id you just retrieved.
4. Summarize the result in plain English. Don't dump JSON.

Rules:
- Never give clinical or medical advice.
- Don't quote dollar prices — defer pricing to a sales contact.
- Don't reveal information about other accounts.
- If the account is on hold, mention it briefly and offer to escalate.
- If a tool returns ``{"found": false}``, ask for a different identifier.
- Keep replies under 120 words.
"""


async def lookup_node(state: State) -> dict[str, Any]:
    chat = get_chat_model("tool", temperature=0.0, max_tokens=600).bind_tools(
        LOOKUP_TOOLS
    )

    history = state.get("messages", [])
    sys = SystemMessage(content=SYSTEM_PROMPT)

    # ``new_messages`` is what we APPEND to state — the AIMessage(s) and any
    # ToolMessage results. The history slice plus these reconstruct each
    # successive prompt.
    new_messages: list[Any] = []
    tool_calls_seen: list[dict[str, Any]] = []
    tool_results_seen: list[dict[str, Any]] = []

    msg = await chat.ainvoke([sys, *history])
    if not isinstance(msg, AIMessage):
        msg = AIMessage(content=str(msg))
    new_messages.append(msg)

    iterations = 0
    while getattr(msg, "tool_calls", None) and iterations < _MAX_TOOL_ITERATIONS:
        iterations += 1
        for tc in msg.tool_calls:
            tool_calls_seen.append({"name": tc.get("name"), "args": tc.get("args")})
            tool_fn = TOOL_MAP.get(tc.get("name", ""))
            if tool_fn is None:
                result_payload: Any = {"error": f"unknown tool: {tc.get('name')}"}
            else:
                try:
                    result_payload = tool_fn.invoke(tc.get("args") or {})
                except Exception as exc:  # surface to the model, not the user
                    result_payload = {"error": f"{type(exc).__name__}: {exc}"}
            tool_results_seen.append(
                {"name": tc.get("name"), "result": result_payload}
            )
            new_messages.append(
                ToolMessage(
                    content=json.dumps(result_payload, default=str),
                    tool_call_id=tc.get("id", ""),
                    name=tc.get("name", ""),
                )
            )

        msg = await chat.ainvoke([sys, *history, *new_messages])
        if not isinstance(msg, AIMessage):
            msg = AIMessage(content=str(msg))
        new_messages.append(msg)

    log.info(
        "lookup.complete",
        iterations=iterations,
        tool_calls=len(tool_calls_seen),
    )

    return {
        "messages": new_messages,
        "tool_calls": tool_calls_seen,
        "tool_results": tool_results_seen,
    }


def _last_user_text(state: State) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""
