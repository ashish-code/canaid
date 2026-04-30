"""Catalog RAG agent — TWO graph nodes.

Why two nodes (retrieve → generate) instead of one:

  1. **Citations arrive before tokens.** The UI can render source cards
     during the `generate` stream — the SSE timeline shows retrieve →
     citations → tokens, which feels right and gives the user visible
     proof the answer is grounded.
  2. **Pure-Python retrieve is independently testable.** No LLM mocking
     required to verify "did we retrieve the right chunk for this query".
  3. **Independent observability.** Each node gets its own LangFuse span
     in Phase 8 — retrieve latency vs. generate latency are separate
     dials, not one number.

The generate node uses `ChatBedrockConverse` so its tokens flow through
the existing `astream_events` plumbing. The system prompt instructs it to
cite passages by their numeric ID — the API maps these back to `doc_id`
in the citations frame.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from canaid.graph.state import State
from canaid.llm.lc import get_chat_model
from canaid.retrieval.retriever import get_retriever

log = structlog.get_logger(__name__)


def _last_user_text(state: State) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def rag_retrieve_node(state: State) -> dict[str, Any]:
    query = _last_user_text(state)
    if not query:
        return {"citations": []}

    retriever = get_retriever()
    chunks = retriever.search(query, k=5)
    citations = [
        {
            "id": i + 1,
            "doc_id": c.doc_id,
            "doc_type": c.doc_type,
            "title": c.title,
            "similarity": round(c.similarity, 3),
            "content": c.content,
            "metadata": c.metadata,
        }
        for i, c in enumerate(chunks)
    ]
    log.info("rag.retrieve", query_chars=len(query), hits=len(citations))
    return {"citations": citations}


SYSTEM_PROMPT = """You are HealthSupplyCo's catalog assistant. Answer the user's \
question using ONLY the numbered passages below. Each passage is a SKU card \
or policy excerpt.

Rules:
- Cite every factual claim by passage number in square brackets, e.g., [2].
- If the passages don't cover the question, say so plainly. Do not guess.
- Do not quote prices. If asked, say a sales contact can prepare a quote.
- Never give clinical or medical advice.
- Keep replies concise — under 150 words unless the user asked for detail.
"""


def _format_passages(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "(no relevant passages were retrieved)"
    lines: list[str] = []
    for c in citations:
        lines.append(f"[{c['id']}] {c['title']} ({c['doc_id']}, {c['doc_type']})")
        lines.append(c["content"])
        lines.append("")
    return "\n".join(lines)


async def rag_generate_node(state: State) -> dict[str, Any]:
    citations: list[dict[str, Any]] = state.get("citations") or []
    if not citations:
        # Honest fallback: tell the user we have nothing to cite.
        msg = AIMessage(
            content=(
                "I couldn't find anything in our catalog that matches your "
                "question. Could you rephrase, or shall I connect you with a "
                "teammate who can dig further?"
            )
        )
        return {"messages": [msg]}

    history = state.get("messages", [])
    user_text = _last_user_text(state)

    # Compose a fresh "user" turn that bundles the retrieved passages with the
    # original question. This isolates the RAG context from earlier turns and
    # keeps the system prompt small + cacheable.
    composed_user = HumanMessage(
        content=(
            f"Question: {user_text}\n\n"
            f"Passages:\n{_format_passages(citations)}"
        )
    )

    chat = get_chat_model("rag", temperature=0.2, max_tokens=600)
    response = await chat.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), *history[:-1], composed_user]
    )
    return {"messages": [response]}
