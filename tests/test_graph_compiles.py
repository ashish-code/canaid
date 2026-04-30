"""Verify the LangGraph workflow compiles cleanly. No AWS calls."""

from __future__ import annotations

from canaid.graph.workflow import RESPONSE_NODES, build_graph, get_graph


def test_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_get_graph_is_cached() -> None:
    assert get_graph() is get_graph()


def test_response_nodes_known() -> None:
    assert "qualifier" in RESPONSE_NODES
    assert "fallback" in RESPONSE_NODES
    assert "rag_generate" in RESPONSE_NODES
    # The intent classifier must NOT be in here — it uses tool-use, not streaming.
    assert "intent" not in RESPONSE_NODES
    # rag_retrieve does no LLM call — must not be a streaming node either.
    assert "rag_retrieve" not in RESPONSE_NODES


def test_graph_has_expected_nodes() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__", "__end__",
        "intent",
        "qualifier",
        "rag_retrieve", "rag_generate",
        "lookup",
        "escalation", "refusal", "fallback",
    }
    assert expected <= nodes, f"missing: {expected - nodes}"


def test_lookup_in_tool_using_nodes() -> None:
    from canaid.graph.workflow import TOOL_USING_NODES
    assert "lookup" in TOOL_USING_NODES
    # lookup is a tool-loop node, not a streaming-token node.
    assert "lookup" not in RESPONSE_NODES
