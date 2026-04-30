"""Smoke tests — no AWS calls, no network."""

from canaid.config import get_settings
from canaid.llm.registry import get_all_specs, get_model_spec


def test_settings_defaults() -> None:
    s = get_settings()
    assert s.api_port > 0
    assert s.aws_region


def test_model_registry_covers_all_agents() -> None:
    specs = get_all_specs()
    agents = {s.agent for s in specs}
    assert agents == {
        "supervisor", "intent", "qualifier", "rag", "tool", "summarizer"
    }


def test_each_spec_has_a_model_and_vendor() -> None:
    for s in get_all_specs():
        assert s.model_id, f"{s.agent} missing model_id"
        assert s.vendor != "unknown", f"{s.agent} unknown vendor for {s.model_id}"


def test_supervisor_is_a_reasoning_model() -> None:
    spec = get_model_spec("supervisor")
    # Reasoning seat should land on a high-tier, tool-capable model.
    assert spec.cost_tier == "high"
    assert spec.supports_tools


def test_intent_is_cheap() -> None:
    spec = get_model_spec("intent")
    assert spec.cost_tier == "low"


def test_we_use_more_than_one_vendor() -> None:
    """The whole point of per-agent models is multi-vendor portability."""
    vendors = {s.vendor for s in get_all_specs()}
    assert len(vendors) >= 2, vendors
