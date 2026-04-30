"""LangGraph workflow definition.

```
            ┌─────────┐
            │  intent │   (Haiku, native Bedrock toolConfig)
            └────┬────┘
                 │ supervisor_route(state) — pure function
                 ▼
   ┌────────┬────────┬────────┬────────┬────────┬─────────┐
   │qualif. │  rag   │ lookup │escalat.│refusal │ fallback│
   │(Sonnet)│(stub→3)│(stub→4)│(Nova)  │(Nova)  │(Sonnet) │
   └────┬───┴────┬───┴────┬───┴────┬───┴────┬───┴────┬────┘
        ▼        ▼        ▼        ▼        ▼        ▼
                              END

```

Compiled with an in-memory checkpointer for Phase 2; Phase 6 swaps in the
Postgres / DynamoDB checkpointer so conversation state survives restarts
and rolling deploys.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from canaid.agents.fallback import fallback_node
from canaid.agents.intent import intent_node
from canaid.agents.lookup import lookup_node
from canaid.agents.qualifier import qualifier_node
from canaid.agents.rag import rag_generate_node, rag_retrieve_node
from canaid.agents.stubs import (
    escalation_node,
    refusal_node,
)
from canaid.graph.router import supervisor_route
from canaid.graph.state import State
from canaid.memory.checkpointer import get_checkpointer

# Names of nodes whose LLM token streams should be forwarded to the user.
# The intent node uses Bedrock's tool-use (non-streaming) — its partial JSON
# must not leak. `rag_retrieve` does no LLM call at all.
# `lookup` runs a tool loop and emits its own SSE frames (tool_call /
# tool_result + a final assistant message), so it's not a streaming-token
# node either.
RESPONSE_NODES: frozenset[str] = frozenset({
    "qualifier",
    "rag_generate",
    "escalation",
    "refusal",
    "fallback",
})

# Nodes that emit citations into state. The API uses this to decide when
# to fire a `citations` SSE frame.
CITATION_EMITTING_NODES: frozenset[str] = frozenset({"rag_retrieve"})

# Nodes that run a tool-use loop. The API streams tool_call / tool_result
# events for any LangChain tool invoked while one of these is active.
TOOL_USING_NODES: frozenset[str] = frozenset({"lookup"})


def build_graph():
    g: StateGraph = StateGraph(State)

    g.add_node("intent", intent_node)
    g.add_node("qualifier", qualifier_node)
    g.add_node("rag_retrieve", rag_retrieve_node)
    g.add_node("rag_generate", rag_generate_node)
    g.add_node("lookup", lookup_node)
    g.add_node("escalation", escalation_node)
    g.add_node("refusal", refusal_node)
    g.add_node("fallback", fallback_node)

    g.set_entry_point("intent")
    g.add_conditional_edges(
        "intent",
        supervisor_route,
        {
            "qualifier": "qualifier",
            "rag": "rag_retrieve",          # retrieve → generate
            "lookup": "lookup",
            "escalation": "escalation",
            "refusal": "refusal",
            "fallback": "fallback",
        },
    )
    g.add_edge("rag_retrieve", "rag_generate")
    for terminal in (
        "qualifier", "rag_generate", "lookup", "escalation", "refusal", "fallback"
    ):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=get_checkpointer())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
