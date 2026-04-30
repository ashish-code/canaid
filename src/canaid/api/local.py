"""In-process chat runner — produces the same frames as `/chat/stream`.

The FastAPI handler wraps these frames in SSE (Server-Sent Events).
The Streamlit-embedded path consumes them directly. This file is the
single source of truth for the frame protocol — both paths converge
through it.

Frame shapes (all plain dicts, JSON-serializable):

    {"type": "cache_hit"}
    {"type": "intent",     "data": {intent, user_type, confidence, rationale, route}}
    {"type": "agent_start","data": {"name": "qualifier"}}
    {"type": "citations",  "data": [{id, doc_id, doc_type, title, similarity}, ...]}
    {"type": "tool_call",  "data": {"name": ..., "args": ...}}
    {"type": "tool_result","data": {"name": ..., "output": ...}}
    {"type": "token",      "data": "...delta..."}
    {"type": "usage",      "data": {input_tokens, output_tokens, cost_usd, latency_ms, by_model}}
    {"type": "done",       "conversation_id": "..."}
    {"type": "error",      "message": "..."}
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from canaid.cache import get_turn_cache
from canaid.graph.router import supervisor_route
from canaid.graph.workflow import (
    CITATION_EMITTING_NODES,
    RESPONSE_NODES,
    TOOL_USING_NODES,
    get_graph,
)
from canaid.observability import (
    AuditEvent,
    cost_for,
    get_audit_writer,
    get_langfuse_handler,
)
from canaid.observability.logging import (
    bind_request_context,
    clear_request_context,
)

log = structlog.get_logger(__name__)


def _extract_token_text(content: Any) -> str:
    """ChatBedrockConverse stream chunks come as either a plain string or a
    list of content blocks. Pull just the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


async def run_chat(
    message: str,
    *,
    conversation_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Drive the LangGraph workflow for one user turn, yielding frames.

    `conversation_id is None` triggers cache lookup + cache write (we
    only cache fresh threads — see `docs/06-caching-memory.md` for the
    rationale).
    """
    conv_id = conversation_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    bind_request_context(request_id=request_id, conversation_id=conv_id)
    log.info("chat.request", message_chars=len(message))

    cache = get_turn_cache()
    audit = get_audit_writer()
    langfuse = get_langfuse_handler()
    t0 = time.perf_counter()

    cached_frames = cache.get(message) if conversation_id is None else None

    graph = get_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": conv_id}}
    if langfuse is not None:
        config["callbacks"] = [langfuse]
    input_state = {"messages": [HumanMessage(content=message)]}

    active_response_node: str | None = None
    frames_to_cache: list[dict[str, Any]] = []
    intent_seen: str | None = None
    user_type_seen: str | None = None
    confidence_seen: float | None = None
    route_seen: str | None = None
    accumulated_response_text: list[str] = []
    usage_total: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }

    def _capture(frame: dict[str, Any]) -> None:
        if frame.get("type") in {
            "intent", "agent_start", "citations",
            "tool_call", "tool_result", "token",
        }:
            frames_to_cache.append(frame)

    try:
        if cached_frames is not None:
            yield {"type": "cache_hit"}
            for frame in cached_frames:
                yield frame
            yield {"type": "done", "conversation_id": conv_id}
            return

        async for ev in graph.astream_events(
            input_state, config=config, version="v2"
        ):
            kind = ev["event"]
            name = ev.get("name", "")
            metadata = ev.get("metadata", {})

            # 1. Intent node finished — classification + route.
            if kind == "on_chain_end" and name == "intent":
                out = ev["data"].get("output") or {}
                if isinstance(out, dict) and out:
                    route = supervisor_route({**out})  # type: ignore[arg-type]
                    intent_seen = out.get("intent")
                    user_type_seen = out.get("user_type")
                    confidence_seen = out.get("confidence")
                    route_seen = route
                    log.info(
                        "chat.routed",
                        intent=intent_seen,
                        user_type=user_type_seen,
                        confidence=confidence_seen,
                        route=route,
                    )
                    frame = {
                        "type": "intent",
                        "data": {
                            "intent": out.get("intent"),
                            "user_type": out.get("user_type"),
                            "confidence": out.get("confidence"),
                            "rationale": out.get("rationale"),
                            "route": route,
                        },
                    }
                    _capture(frame)
                    yield frame

            # 2. Citation-emitting node finished — citations frame.
            if kind == "on_chain_end" and name in CITATION_EMITTING_NODES:
                out = ev["data"].get("output") or {}
                citations = (out or {}).get("citations") or []
                wire = [
                    {
                        "id": c.get("id"),
                        "doc_id": c.get("doc_id"),
                        "doc_type": c.get("doc_type"),
                        "title": c.get("title"),
                        "similarity": c.get("similarity"),
                    }
                    for c in citations
                ]
                frame = {"type": "citations", "data": wire}
                _capture(frame)
                yield frame

            # 3. Response or tool-using node started — announce.
            if (
                kind == "on_chain_start"
                and name in (RESPONSE_NODES | TOOL_USING_NODES)
                and active_response_node != name
            ):
                active_response_node = name
                frame = {"type": "agent_start", "data": {"name": name}}
                _capture(frame)
                yield frame

            # 4. Tool calls (lookup agent).
            if kind == "on_tool_start" and metadata.get(
                "langgraph_node", ""
            ) in TOOL_USING_NODES:
                frame = {
                    "type": "tool_call",
                    "data": {"name": name, "args": ev["data"].get("input")},
                }
                _capture(frame)
                yield frame
            if kind == "on_tool_end" and metadata.get(
                "langgraph_node", ""
            ) in TOOL_USING_NODES:
                out = ev["data"].get("output")
                wire_out = out
                if isinstance(out, dict) and "orders" in out:
                    wire_out = {
                        "orders_count": len(out.get("orders") or []),
                        "first_order_id": (out.get("orders") or [{}])[0].get(
                            "order_id"
                        ),
                    }
                frame = {
                    "type": "tool_result",
                    "data": {"name": name, "output": wire_out},
                }
                _capture(frame)
                yield frame

            # 5. LLM stream token from a streaming response node.
            if kind == "on_chat_model_stream":
                node = metadata.get("langgraph_node", "")
                if node in RESPONSE_NODES:
                    chunk = ev["data"]["chunk"]
                    content = getattr(chunk, "content", chunk)
                    text = _extract_token_text(content)
                    if text:
                        accumulated_response_text.append(text)
                        frame = {"type": "token", "data": text}
                        _capture(frame)
                        yield frame

            # 5b. LLM call ended — accumulate cost.
            if kind == "on_chat_model_end":
                usage = (ev.get("data") or {}).get("usage") or {}
                msg_out = (ev.get("data") or {}).get("output")
                if msg_out is not None:
                    um = getattr(msg_out, "usage_metadata", None)
                    if um:
                        usage = {
                            "inputTokens": um.get("input_tokens", 0),
                            "outputTokens": um.get("output_tokens", 0),
                        }
                in_tok = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
                out_tok = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
                meta = ev.get("metadata") or {}
                model_id = (
                    meta.get("ls_model_name")
                    or meta.get("model")
                    or meta.get("model_id")
                    or "unknown"
                )
                call_cost = cost_for(model_id, in_tok, out_tok)
                usage_total["input_tokens"] += in_tok
                usage_total["output_tokens"] += out_tok
                usage_total["cost_usd"] += call_cost
                m = usage_total["by_model"].setdefault(
                    model_id,
                    {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
                )
                m["input_tokens"] += in_tok
                m["output_tokens"] += out_tok
                m["cost_usd"] += call_cost

            # 6. Lookup node finished — emit final assistant as one token frame.
            if kind == "on_chain_end" and name in TOOL_USING_NODES:
                out = ev["data"].get("output") or {}
                msgs = (out or {}).get("messages") or []
                if msgs:
                    last = msgs[-1]
                    final_text = getattr(last, "content", "")
                    if isinstance(final_text, list):
                        final_text = _extract_token_text(final_text)
                    if isinstance(final_text, str) and final_text:
                        frame = {"type": "token", "data": final_text}
                        _capture(frame)
                        yield frame

        # Cache writeback (only first-turn fresh threads).
        if conversation_id is None:
            cache.set_if_cacheable(
                message, frames_to_cache, intent=intent_seen
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_wire = {
            "input_tokens": usage_total["input_tokens"],
            "output_tokens": usage_total["output_tokens"],
            "cost_usd": round(usage_total["cost_usd"], 6),
            "latency_ms": latency_ms,
            "by_model": {
                m: {
                    "input_tokens": v["input_tokens"],
                    "output_tokens": v["output_tokens"],
                    "cost_usd": round(v["cost_usd"], 6),
                }
                for m, v in usage_total["by_model"].items()
            },
        }
        yield {"type": "usage", "data": usage_wire}

        audit.write(
            AuditEvent(
                request_id=request_id,
                conversation_id=conv_id,
                user_message_redacted=message,
                intent=intent_seen,
                user_type=user_type_seen,
                confidence=confidence_seen,
                route=route_seen,
                response_redacted="".join(accumulated_response_text)[:4000],
                input_tokens=usage_total["input_tokens"],
                output_tokens=usage_total["output_tokens"],
                cost_usd=round(usage_total["cost_usd"], 6),
                latency_ms=latency_ms,
                metadata={"by_model": usage_wire["by_model"]},
            )
        )

        yield {"type": "done", "conversation_id": conv_id}
    except Exception as exc:
        log.exception("chat.error")
        yield {
            "type": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }
    finally:
        clear_request_context()
