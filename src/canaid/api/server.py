"""FastAPI app — public HTTP surface.

Endpoints:
  GET  /health        liveness probe
  GET  /models        which model each agent is using right now (debug aid)
  POST /chat/stream   stream a reply via Server-Sent Events

The streaming endpoint is a thin SSE wrapper around `canaid.api.local.run_chat`,
which is the in-process source of truth for the chat protocol. The same
async generator powers the embedded (Streamlit Cloud) deployment.

SSE frame shapes are documented in `canaid/api/local.py`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.cors import CORSMiddleware

from canaid.api.local import run_chat
from canaid.llm.registry import get_all_specs
from canaid.observability.logging import configure_logging

configure_logging()
log = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


class ModelInfo(BaseModel):
    agent: str
    model_id: str
    vendor: str
    cost_tier: str


def _sse(payload: dict[str, Any]) -> dict[str, str]:
    return {"data": json.dumps(payload)}


def create_app() -> FastAPI:
    app = FastAPI(
        title="CanAID API",
        version="0.2.0",
        description=(
            "Multi-agent contact-center chatbot harness. The /chat/stream "
            "endpoint wraps `canaid.api.local.run_chat` in SSE."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        return [
            ModelInfo(
                agent=s.agent,
                model_id=s.model_id,
                vendor=s.vendor,
                cost_tier=s.cost_tier,
            )
            for s in get_all_specs()
        ]

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        async def event_gen() -> AsyncIterator[dict[str, str]]:
            async for frame in run_chat(
                req.message, conversation_id=req.conversation_id
            ):
                yield _sse(frame)

        return EventSourceResponse(event_gen())

    return app


app = create_app()
