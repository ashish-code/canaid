"""Router unit tests — pure-Python, no AWS calls.

The router is the single most-evaluated component of the harness because
every other agent's eval starts from "did we route here in the first place?"
"""

from __future__ import annotations

import pytest

from canaid.graph.router import supervisor_route


@pytest.mark.parametrize(
    ("intent", "confidence", "expected"),
    [
        ("lead_qualification", 0.9, "qualifier"),
        ("catalog_question", 0.9, "rag"),
        ("account_lookup", 0.9, "lookup"),
        ("escalation", 0.4, "escalation"),   # escalation always routes through
        ("refusal", 0.4, "refusal"),          # refusal always routes through
        ("small_talk", 0.9, "fallback"),
        ("unknown", 0.99, "fallback"),
        # Low-confidence specialist intents fall back to general assistant.
        ("lead_qualification", 0.3, "fallback"),
        ("catalog_question", 0.4, "fallback"),
        ("account_lookup", 0.5, "fallback"),
    ],
)
def test_supervisor_route(intent: str, confidence: float, expected: str) -> None:
    state = {"intent": intent, "confidence": confidence}
    assert supervisor_route(state) == expected  # type: ignore[arg-type]


def test_supervisor_route_handles_missing_fields() -> None:
    assert supervisor_route({}) == "fallback"
    assert supervisor_route({"intent": "small_talk"}) == "fallback"


def test_supervisor_route_handles_unknown_intent_label() -> None:
    """An intent label not in the table — defensive default."""
    state = {"intent": "weird_new_thing", "confidence": 0.9}
    assert supervisor_route(state) == "fallback"  # type: ignore[arg-type]
