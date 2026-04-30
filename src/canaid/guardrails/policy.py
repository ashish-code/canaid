"""Bedrock Guardrails wiring.

We don't *create* the guardrail at runtime — that's a control-plane action
done once via `scripts/setup_guardrail.py` (or the Phase 9 CDK stack).
Once created, both the raw `BedrockClient` and `ChatBedrockConverse` send
the guardrail identifier on every Converse call:

```python
converse(modelId=..., guardrailConfig={
    "guardrailIdentifier": "abcd1234",
    "guardrailVersion": "1",
})
```

If `CANAID_GUARDRAIL_ID` isn't set in the environment, we no-op — the bot
runs without policy enforcement at the LLM boundary. Useful in dev when you
don't want to pay for guardrail invocations.
"""

from __future__ import annotations

from canaid.config import get_settings


def guardrail_config_for_converse() -> dict[str, str] | None:
    """Return the kwargs to merge into `client.converse(...)` — or None."""
    s = get_settings()
    if not s.guardrail_id:
        return None
    return {
        "guardrailIdentifier": s.guardrail_id,
        "guardrailVersion": s.guardrail_version,
    }


def guardrails_for_chat_bedrock() -> dict[str, str] | None:
    """Return the kwargs `langchain_aws.ChatBedrockConverse` expects via
    its `guardrails=` parameter — or None."""
    s = get_settings()
    if not s.guardrail_id:
        return None
    return {
        "guardrailIdentifier": s.guardrail_id,
        "guardrailVersion": s.guardrail_version,
    }
