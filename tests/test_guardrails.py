"""Guardrail unit tests — pure-Python, no AWS calls."""

from __future__ import annotations

from canaid.guardrails.log_filter import (
    pii_log_processor,
    regex_redact,
)
from canaid.guardrails.pii import PiiEntity, redact_with_entities
from canaid.guardrails.refusals import REFUSAL_REASONS, refusal_text


def test_regex_redact_email() -> None:
    out = regex_redact("contact me at marc.tremblay@riverdalegh.example today")
    assert "marc.tremblay" not in out
    assert "[EMAIL]" in out


def test_regex_redact_phone() -> None:
    for variant in [
        "call +1-416-555-0142",
        "call 416-555-0142",
        "call (416) 555-0142",
        "call 14165550142",
    ]:
        out = regex_redact(variant)
        assert "555-0142" not in out
        assert "[PHONE]" in out, f"missed phone in: {variant}"


def test_regex_redact_card() -> None:
    out = regex_redact("card 4111 1111 1111 1111 expiring soon")
    assert "4111" not in out
    assert "[CARD]" in out


def test_regex_redact_idempotent() -> None:
    msg = "ping marc.tremblay@riverdalegh.example or 416-555-0142"
    once = regex_redact(msg)
    twice = regex_redact(once)
    assert once == twice


def test_log_processor_skips_safe_keys() -> None:
    event = {
        "event": "chat.request",
        "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "user_text": "I'm priya.desai@northbridgefh.example, please help.",
    }
    out = pii_log_processor(None, "info", dict(event))
    # safe key passes through untouched.
    assert out["model_id"] == event["model_id"]
    # PII key gets scrubbed.
    assert "priya.desai" not in out["user_text"]
    assert "[EMAIL]" in out["user_text"]


def test_log_processor_handles_non_string_values() -> None:
    out = pii_log_processor(None, "info", {"event": "x", "n": 42, "ok": True})
    assert out == {"event": "x", "n": 42, "ok": True}


def test_redact_with_entities_walks_in_reverse() -> None:
    text = "name Priya Desai email priya@example.com"
    entities = [
        PiiEntity(type="NAME", start=5, end=16, score=0.99),
        PiiEntity(type="EMAIL", start=23, end=40, score=0.99),
    ]
    out = redact_with_entities(text, entities)
    assert "Priya Desai" not in out
    assert "priya@example.com" not in out
    assert "[NAME]" in out
    assert "[EMAIL]" in out


def test_refusal_templates_cover_known_reasons() -> None:
    for reason in REFUSAL_REASONS:
        text = refusal_text(reason)
        assert isinstance(text, str)
        assert len(text) > 30


def test_refusal_falls_back_for_unknown_reason() -> None:
    assert refusal_text("totally_made_up_reason") == refusal_text("unknown")


def test_refusal_for_clinical_advice_offers_clinician() -> None:
    txt = refusal_text("clinical_advice")
    assert "clinic" in txt.lower() or "medical" in txt.lower()
