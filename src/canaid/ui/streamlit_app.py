"""Streamlit chat UI.

Two modes selected at startup:

  * **API mode** — `CANAID_API_URL` is set; the UI streams via httpx/SSE
    from the FastAPI backend. Production-style separation.
  * **Embedded mode** — `CANAID_API_URL` is unset/blank; the UI runs the
    LangGraph workflow in-process via `canaid.api.local.run_chat`. Used
    for the Streamlit Community Cloud deploy where we can't host a
    sidecar service.

Both modes consume the *same frame protocol* (see `canaid/api/local.py`).
The render code below is shared.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from typing import Any

import streamlit as st

API_URL = os.getenv("CANAID_API_URL", "").strip()
EMBEDDED_MODE = not API_URL


_AGENT_LABELS = {
    "qualifier": "Lead Qualifier",
    "rag_generate": "Catalog RAG",
    "lookup": "Account Lookup",
    "escalation": "Escalation",
    "refusal": "Policy Refusal",
    "fallback": "General Assistant",
}


# ---- Streaming bridges -----------------------------------------------------
def _api_stream(message: str, conversation_id: str | None) -> Iterator[dict[str, Any]]:
    """Stream frames from the FastAPI /chat/stream endpoint over SSE."""
    import httpx  # local import — Streamlit Cloud doesn't need it in embedded mode

    with httpx.stream(
        "POST",
        f"{API_URL}/chat/stream",
        json={"message": message, "conversation_id": conversation_id},
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            yield json.loads(line[len("data:"):].strip())


def _embedded_stream(
    message: str, conversation_id: str | None
) -> Iterator[dict[str, Any]]:
    """Drive `run_chat` (async generator) from sync Streamlit code.

    Streamlit's render loop is sync; we own one event loop, drive the async
    generator one frame at a time, and yield to the caller in between so
    Streamlit can repaint mid-turn.
    """
    from canaid.api.local import run_chat

    loop = asyncio.new_event_loop()
    try:
        agen = run_chat(message, conversation_id=conversation_id)
        while True:
            try:
                frame = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
            yield frame
    finally:
        loop.close()


def _frame_stream(message: str, conversation_id: str | None) -> Iterator[dict[str, Any]]:
    if EMBEDDED_MODE:
        yield from _embedded_stream(message, conversation_id)
    else:
        yield from _api_stream(message, conversation_id)


# ---- Streamlit Cloud secret bridge -----------------------------------------
def _bridge_streamlit_secrets() -> None:
    """Streamlit Cloud injects secrets via `st.secrets`; copy each entry into
    os.environ so our pydantic Settings + boto3's default chain pick them up
    without a Streamlit-aware code path."""
    try:
        for k in list(st.secrets):  # type: ignore[union-attr]
            v = st.secrets[k]
            if isinstance(v, str) and not os.getenv(k):
                os.environ[k] = v
    except (FileNotFoundError, KeyError, AttributeError):
        # No secrets file (local dev) — nothing to do.
        return


_bridge_streamlit_secrets()


# ---- Page ------------------------------------------------------------------
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
    st.session_state.messages = []
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
    mode_label = "embedded (single-process)" if EMBEDDED_MODE else f"API @ {API_URL}"
    st.caption(f"**Mode:** {mode_label}")

    st.subheader("Models per agent")
    if EMBEDDED_MODE:
        # No API to query — render directly from the registry.
        from canaid.llm.registry import get_all_specs

        for s in get_all_specs():
            st.markdown(
                f"**{s.agent}** — `{s.model_id}`  \n"
                f"<sub>{s.vendor} · {s.cost_tier} cost</sub>",
                unsafe_allow_html=True,
            )
    else:
        try:
            import httpx
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


# ---- Render helpers --------------------------------------------------------
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


def _render_usage(usage: dict | None) -> None:
    if not usage:
        return
    with st.expander("usage / cost", expanded=False):
        st.markdown(
            f"**tokens:** {usage.get('input_tokens', 0)} in / "
            f"{usage.get('output_tokens', 0)} out  \n"
            f"**latency:** {usage.get('latency_ms', 0)} ms  \n"
            f"**cost:** ${usage.get('cost_usd', 0):.6f}"
        )
        bym = usage.get("by_model") or {}
        if bym:
            st.markdown("**by model:**")
            for model_id, v in bym.items():
                st.markdown(
                    f"  · `{model_id}` — {v.get('input_tokens', 0)} in / "
                    f"{v.get('output_tokens', 0)} out · ${v.get('cost_usd', 0):.6f}"
                )


# ---- History ---------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("agent trace", expanded=False):
                _render_trace(msg["trace"])
        if msg["role"] == "assistant" and msg.get("citations"):
            _render_citations(msg["citations"])
        if msg["role"] == "assistant" and msg.get("tool_events"):
            _render_tool_calls(msg["tool_events"])
        if msg["role"] == "assistant" and msg.get("usage"):
            _render_usage(msg["usage"])
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
        cache_box = st.empty()
        trace_box = st.empty()
        cite_box = st.empty()
        tool_box = st.empty()
        body = st.empty()
        usage_box = st.empty()

        accumulated = ""
        trace: dict = {}
        citations: list[dict] = []
        tool_events: list[dict] = []
        usage: dict = {}
        cache_hit = False

        try:
            for payload in _frame_stream(prompt, st.session_state.conversation_id):
                kind = payload.get("type")
                if kind == "cache_hit":
                    cache_hit = True
                    cache_box.success("⚡ cache hit — replayed without LLM call")
                elif kind == "intent":
                    trace.update(payload.get("data") or {})
                    with trace_box.container(), st.expander(
                        "agent trace", expanded=True
                    ):
                        _render_trace(trace)
                elif kind == "agent_start":
                    trace["agent"] = (payload.get("data") or {}).get("name")
                    with trace_box.container(), st.expander(
                        "agent trace", expanded=True
                    ):
                        _render_trace(trace)
                elif kind == "citations":
                    citations = payload.get("data") or []
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
                    accumulated += payload.get("data") or ""
                    body.markdown(accumulated + "▌")
                elif kind == "usage":
                    usage = payload.get("data") or {}
                    with usage_box.container():
                        _render_usage(usage)
                elif kind == "done":
                    st.session_state.conversation_id = payload.get(
                        "conversation_id"
                    )
                elif kind == "error":
                    accumulated = f"_Error: {payload.get('message')}_"
        except Exception as exc:
            accumulated = f"_Stream error: {type(exc).__name__}: {exc}_"

        body.markdown(accumulated or "_(no response)_")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": accumulated,
                "trace": trace,
                "citations": citations,
                "tool_events": tool_events,
                "usage": usage,
                "cache_hit": cache_hit,
            }
        )
