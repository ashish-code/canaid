"""Templated refusals.

When the bot must decline, the user should see consistent language regardless
of *which layer* refused (Bedrock Guardrails policy, intent-classifier hit,
or our own routing fallback). Inconsistent refusal voice is a tell that
something is brittle.

Each template ends with a "would you like a teammate?" off-ramp so the user
always has an obvious next step. Compliance and trust both improve when
"no" comes with a path forward.
"""

from __future__ import annotations

from typing import Final

REFUSAL_REASONS: Final[set[str]] = {
    "clinical_advice",
    "pricing",
    "pii_input",
    "off_topic",
    "jailbreak",
    "policy",
    "unknown",
}


_TEMPLATES: dict[str, str] = {
    "clinical_advice": (
        "I can't provide clinical or medical advice — that needs to come "
        "from a clinician. Want me to connect you with someone on our team "
        "who can help with the supply-side of your question?"
    ),
    "pricing": (
        "I'm not able to quote specific prices in this channel. Your sales "
        "contact can prepare a quote in a day — should I have someone reach "
        "out?"
    ),
    "pii_input": (
        "It looks like your message contains personal information we shouldn't "
        "handle here. Could you rephrase, or shall I escalate to a teammate "
        "who can take that securely?"
    ),
    "off_topic": (
        "I'm focused on healthcare-supply-chain questions for HealthSupplyCo "
        "clients. Is there something on that side I can help with?"
    ),
    "jailbreak": (
        "I can't help with that. If there's something else on the supply or "
        "account side I can help with, I'm happy to."
    ),
    "policy": (
        "That falls outside what I can help with directly. Want me to connect "
        "you with a teammate?"
    ),
    "unknown": (
        "I'm sorry, I can't help with that one. Want me to connect you with a "
        "teammate?"
    ),
}


def refusal_text(reason: str | None) -> str:
    return _TEMPLATES.get(reason or "unknown", _TEMPLATES["unknown"])
