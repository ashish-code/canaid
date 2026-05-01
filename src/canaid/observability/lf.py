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
    # LangFuse 4.x relocated the LangChain integration; 2.x had it at
    # `langfuse.callback`. Try the new path first, fall back to the old.
    CallbackHandler = None
    try:
        from langfuse.langchain import CallbackHandler  # 3.x / 4.x
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # 2.x
        except ImportError as exc:
            log.warning("langfuse.import_failed", error=str(exc))
            return None
    try:
        # In 4.x the SDK reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from
        # the environment by itself; the callback constructor takes no auth
        # kwargs. We promote keys to os.environ via load_dotenv at startup,
        # so this just works in both local and Streamlit Cloud.
        handler = CallbackHandler()
        log.info("langfuse.handler_ready", host=s.langfuse_host)
        return handler
    except Exception as exc:
        log.warning("langfuse.handler_unavailable", error=str(exc))
        return None
