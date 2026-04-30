"""Streamlit chat UI.

Talks to the FastAPI backend over SSE. Renders a per-turn agent trace
(intent + route + which specialist replied) so a reviewer can see the
multi-agent harness working without reading logs.
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("CANAID_API_URL", "http://localhost:8000")


_AGENT_LABELS = {
    "qualifier": "Lead Qualifier",
    "rag_generate": "Catalog RAG",
    "lookup": "Account Lookup",
    "escalation": "Escalation",
    "refusal": "Policy Refusal",
    "fallback": "General Assistant",
}


st.set_page_config(
    page_title="CanAID — Contact Center",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("CanAID — Contact Center Assistant")
st.caption(
    "Multi-agent chatbot for healthcare supply-chain B2B. "
    "Demo build — no real client data."
)

if "messages" not in st.session_state:
    st.session_state.messages = []           # list[{role, content, trace?, citations?}]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None


_SAMPLE_QUESTIONS = [
    ("New prospect (Lead Qualifier)",
     "Hi, I run a small clinic in Toronto and I'm thinking about switching distributors. Can you tell me how onboarding works?"),
    ("Catalog Q&A (RAG with citations)",
     "Do you carry chemo-tested nitrile gloves and what are they certified to?"),
    ("Account lookup (Tool-use)",
     "Hi, I'm Marc Tremblay from Riverdale General. What are my recent orders?"),
    ("Order status (Tool-use)",
     "Can you check the status of order SO-2026-00805 for me?"),
    ("Clinical-advice refusal (Guardrails)",
     "My patient has a stage 3 pressure ulcer. What dressing protocol should I use?"),
    ("Pricing refusal (Guardrails)",
     "What's the unit price for a case of N95 masks?"),
]

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.subheader("Models per agent")
    try:
        models = httpx.get(f"{API_URL}/models", timeout=5).json()
        for m in models:
            st.markdown(
                f"**{m['agent']}** — `{m['model_id']}`  \n"
                f"<sub>{m['vendor']} · {m['cost_tier']} cost</sub>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.warning(f"Cannot reach API at {API_URL}\n\n{e}")

    st.divider()
    st.subheader("Try a sample question")
    for label, prompt in _SAMPLE_QUESTIONS:
        if st.button(label, key=f"sample-{label}", use_container_width=True):
            st.session_state.queued_prompt = prompt
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.queued_prompt = None
        st.rerun()


# ---- History ---------------------------------------------------------------
def _render_trace(trace: dict | None) -> None:
    if not trace:
        return
    cols = st.columns([1, 1, 1, 1])
    cols[0].markdown(f"**user_type**  \n`{trace.get('user_type','-')}`")
    cols[1].markdown(f"**intent**  \n`{trace.get('intent','-')}`")
    conf = trace.get("confidence")
    cols[2].markdown(
        f"**confidence**  \n`{conf:.2f}`" if isinstance(conf, (int, float)) else "**confidence**  \n`-`"
    )
    route = trace.get("route", "-")
    label = _AGENT_LABELS.get(trace.get("agent", ""), trace.get("agent", "-"))
    cols[3].markdown(f"**routed to**  \n`{route}` → {label}")
    if trace.get("rationale"):
        st.caption(f"_rationale: {trace['rationale']}_")


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"sources ({len(citations)})", expanded=False):
        for c in citations:
            sim = c.get("similarity")
            sim_str = f" · sim {sim:.2f}" if isinstance(sim, (int, float)) else ""
            st.markdown(
                f"**[{c.get('id')}]** `{c.get('doc_id')}` — "
                f"{c.get('title')} *({c.get('doc_type')}{sim_str})*"
            )


def _render_tool_calls(tool_events: list[dict]) -> None:
    if not tool_events:
        return
    with st.expander(f"tool calls ({len(tool_events)})", expanded=False):
        for ev in tool_events:
            kind = ev.get("kind")
            if kind == "call":
                args = ev.get("args") or {}
                st.markdown(f"**→ {ev.get('name')}** `{json.dumps(args)}`")
            elif kind == "result":
                out = ev.get("output")
                st.markdown(f"**← {ev.get('name')}** `{json.dumps(out)}`")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("agent trace", expanded=False):
                _render_trace(msg["trace"])
        if msg["role"] == "assistant" and msg.get("citations"):
            _render_citations(msg["citations"])
        if msg["role"] == "assistant" and msg.get("tool_events"):
            _render_tool_calls(msg["tool_events"])
        st.markdown(msg["content"])


# ---- Input -----------------------------------------------------------------
prompt = st.chat_input("How can we help your facility today?")
if prompt is None and st.session_state.queued_prompt is not None:
    prompt = st.session_state.queued_prompt
    st.session_state.queued_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        trace_box = st.empty()
        cite_box = st.empty()
        tool_box = st.empty()
        body = st.empty()
        accumulated = ""
        trace: dict = {}
        citations: list[dict] = []
        tool_events: list[dict] = []
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/chat/stream",
                json={
                    "message": prompt,
                    "conversation_id": st.session_state.conversation_id,
                },
                timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = json.loads(line[len("data:"):].strip())
                    kind = payload.get("type")
                    if kind == "intent":
                        trace.update(payload["data"])
                        with trace_box.container(), st.expander(
                            "agent trace", expanded=True
                        ):
                            _render_trace(trace)
                    elif kind == "agent_start":
                        trace["agent"] = payload["data"]["name"]
                        with trace_box.container(), st.expander(
                            "agent trace", expanded=True
                        ):
                            _render_trace(trace)
                    elif kind == "citations":
                        citations = payload.get("data", []) or []
                        with cite_box.container():
                            _render_citations(citations)
                    elif kind == "tool_call":
                        tool_events.append(
                            {"kind": "call", **(payload.get("data") or {})}
                        )
                        with tool_box.container():
                            _render_tool_calls(tool_events)
                    elif kind == "tool_result":
                        tool_events.append(
                            {"kind": "result", **(payload.get("data") or {})}
                        )
                        with tool_box.container():
                            _render_tool_calls(tool_events)
                    elif kind == "token":
                        accumulated += payload["data"]
                        body.markdown(accumulated + "▌")
                    elif kind == "done":
                        st.session_state.conversation_id = payload.get(
                            "conversation_id"
                        )
                    elif kind == "error":
                        accumulated = f"_Error: {payload.get('message')}_"
        except httpx.HTTPError as e:
            accumulated = f"_Connection error talking to API: {e}_"
        body.markdown(accumulated or "_(no response)_")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": accumulated,
                "trace": trace,
                "citations": citations,
                "tool_events": tool_events,
            }
        )
