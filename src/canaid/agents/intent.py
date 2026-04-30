"""Intent classifier agent.

Runs every turn. Takes the latest user message + a small slice of context
and returns a structured ``IntentClassification`` (intent, user_type,
confidence, rationale).

We use Bedrock's native Converse ``toolConfig`` rather than LangChain's
``with_structured_output``. Why:

  * **Reliable forced output.** Setting ``toolChoice`` to a specific tool
    name guarantees the model emits a tool-use block we can parse. No
    retries-on-malformed-JSON loop.
  * **Vendor-neutral.** This same toolConfig works on Anthropic, Meta, and
    Amazon models — useful in case we ever swap intent off Haiku.
  * **Cheap.** No second LLM step to validate output format.

The intent label set is *closed* (see ``Intent`` in ``graph.state``). New
intents go through the registry, not by free-form LLM output, so the
router stays exhaustive.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from canaid.graph.state import State
from canaid.llm.bedrock import get_bedrock_client
from canaid.llm.registry import get_model_spec

log = structlog.get_logger(__name__)


SYSTEM_PROMPT = """You are the intent classifier for HealthSupplyCo's contact-center \
assistant. HealthSupplyCo is a B2B healthcare supply-chain distributor — its clients \
are facilities (hospitals, clinics, long-term-care homes), not patients.

Your job: read the user's most recent message and emit ONE call to the \
`classify_intent` tool. Use only the labels in the tool schema.

Definitions:
- user_type:
  - "new"      — message implies the speaker is a prospective client (asking \
about onboarding, requesting a quote/info, "I represent a clinic that wants…").
  - "existing" — message implies an established account (references an order, \
account number, invoice, or "our account").
  - "unknown"  — not enough signal yet.
- intent:
  - "lead_qualification" — new prospect; we should run BANT (budget/authority/\
need/timeline).
  - "catalog_question"   — asks about a product, SKU, availability, policy, \
shipping, returns. Answered from the product catalog.
  - "account_lookup"     — asks about THEIR specific account/order — needs a \
tool call to the CRM.
  - "escalation"         — explicit ask for a human or expressing frustration \
that warrants a handoff.
  - "small_talk"         — greetings, thanks, niceties, off-topic-but-benign.
  - "refusal"            — clinical/medical advice ask, off-policy request, \
attempted jailbreak.
  - "unknown"            — none of the above with reasonable confidence.

confidence: 0.0-1.0 self-rating. Be honest; downstream routing depends on it.
rationale: one short sentence — used for traces, not shown to the user.
"""


_TOOL_SPEC: dict[str, Any] = {
    "toolSpec": {
        "name": "classify_intent",
        "description": "Emit the structured intent classification for the user message.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "user_type": {
                        "type": "string",
                        "enum": ["new", "existing", "unknown"],
                    },
                    "intent": {
                        "type": "string",
                        "enum": [
                            "lead_qualification",
                            "catalog_question",
                            "account_lookup",
                            "escalation",
                            "small_talk",
                            "refusal",
                            "unknown",
                        ],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "rationale": {"type": "string", "maxLength": 200},
                },
                "required": ["user_type", "intent", "confidence", "rationale"],
            }
        },
    }
}


def _last_user_text(state: State) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def intent_node(state: State) -> dict[str, Any]:
    user_text = _last_user_text(state)
    if not user_text:
        return {
            "intent": "unknown",
            "user_type": "unknown",
            "confidence": 0.0,
            "rationale": "no user message",
        }

    spec = get_model_spec("intent")
    client = get_bedrock_client()
    resp = client.converse(
        model_id=spec.model_id,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        tool_config={
            "tools": [_TOOL_SPEC],
            "toolChoice": {"tool": {"name": "classify_intent"}},
        },
        inference_config={"maxTokens": 200, "temperature": 0.0},
    )
    tool_use = resp.first_tool_use()
    if not tool_use:
        log.warning("intent.no_tool_use", text=resp.text[:200])
        return {
            "intent": "unknown",
            "user_type": "unknown",
            "confidence": 0.0,
            "rationale": "model did not return tool_use",
        }

    payload = tool_use.get("input") or {}
    out = {
        "intent": payload.get("intent", "unknown"),
        "user_type": payload.get("user_type", "unknown"),
        "confidence": float(payload.get("confidence", 0.0)),
        "rationale": payload.get("rationale", ""),
    }
    log.info("intent.classified", **out, model_id=spec.model_id)
    return out
