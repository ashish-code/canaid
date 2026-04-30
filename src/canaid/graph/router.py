"""Supervisor routing logic.

Pure Python — no LLM call. Maps a classified intent (+ user_type +
confidence) to the next node. We chose deterministic routing over an
"LLM supervisor" because:

  1. **Testable.** Routing logic is a function, unit-testable in
     milliseconds. An LLM supervisor would need eval traces.
  2. **Cheap.** The intent step already cost an LLM call; spending a second
     one to ask "where should we route?" is duplication.
  3. **Auditable.** A reviewer reading the source can predict the bot's
     behavior on any classified input.

When deterministic routing breaks down — multiple competing intents that
need arbitration, multi-step plan-and-execute — we'll graduate to an
LLM-driven supervisor. We're not there yet.
"""

from __future__ import annotations

from canaid.graph.state import Intent, Route, State

# Confidence threshold below which we route to fallback rather than risk a
# wrong specialist. Tuned via the eval harness in Phase 7.
_MIN_CONFIDENCE = 0.55


_INTENT_TO_ROUTE: dict[Intent, Route] = {
    "lead_qualification": "qualifier",
    "catalog_question": "rag",
    "account_lookup": "lookup",
    "escalation": "escalation",
    "refusal": "refusal",
    "small_talk": "fallback",
    "unknown": "fallback",
}


def supervisor_route(state: State) -> Route:
    """Decide which node handles this turn."""
    intent: Intent = state.get("intent") or "unknown"  # type: ignore[assignment]
    confidence: float = float(state.get("confidence") or 0.0)

    if confidence < _MIN_CONFIDENCE and intent not in {"refusal", "escalation"}:
        return "fallback"
    return _INTENT_TO_ROUTE.get(intent, "fallback")
