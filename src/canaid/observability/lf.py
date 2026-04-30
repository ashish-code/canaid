"""LangFuse Cloud integration.

Emits one trace per turn — every LLM call in the graph becomes a span.
The trace's `userId` and `sessionId` map to our `request_id` and
`conversation_id` so a customer-success engineer can search by either.

Returns ``None`` from `get_langfuse_handler()` if LangFuse keys aren't
configured — calling code should do `if handler: callbacks.append(handler)`
so the handler is optional.

Phase 9 will gate sampling: in prod we'll trace 10% of traffic + 100% of
errors instead of every turn, to keep the LangFuse bill predictable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog

from canaid.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_langfuse_handler() -> Any | None:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        log.info("langfuse.handler_ready", host=s.langfuse_host)
        return handler
    except Exception as exc:
        log.warning("langfuse.handler_unavailable", error=str(exc))
        return None
