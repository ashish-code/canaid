"""Agents.

Each file is one agent. Phase 1 ships the placeholder `echo` agent only —
its job is to prove end-to-end wiring (Bedrock auth, streaming, FastAPI SSE,
Streamlit). Real agents arrive starting Phase 2:

  * intent       (Phase 2) — classify new-vs-existing + intent label
  * qualifier    (Phase 2) — BANT dialog for new prospects
  * supervisor   (Phase 2) — routes turns, owns state (LangGraph node)
  * rag          (Phase 3) — Catalog RAG with citations
  * lookup       (Phase 4) — tool-use over a mock CRM
  * escalation   (Phase 5) — handoff summarizer
"""
