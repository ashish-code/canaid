"""LangGraph checkpointer factory.

Returns a checkpointer suitable for the current environment:
  * In-memory (default) — fastest; loses state on restart.
  * Postgres-backed — durable; uses the same DSN as the RAG store.

The Postgres path requires ``langgraph-checkpoint-postgres``. On first use
it runs ``setup()`` to create its tables (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`) inside the existing CanAID database.

Falls back gracefully to ``MemorySaver`` if Postgres isn't reachable —
the bot keeps working without durability rather than refusing to start.
That's the right trade-off for a contact-center: better to serve the
conversation in-memory than to error out the user's message.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from langgraph.checkpoint.memory import MemorySaver

from canaid.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer():
    s = get_settings()

    if not s.use_postgres_checkpointer:
        log.info("checkpointer.memory_saver")
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        # `from_conn_string` returns a context manager wrapping a connection
        # pool; for our long-lived process we keep it open for the lifetime
        # of the app. The graph compile step holds a reference.
        cm = PostgresSaver.from_conn_string(s.pg_dsn)
        saver = cm.__enter__()
        try:
            saver.setup()
        except Exception as exc:
            log.warning("checkpointer.setup_failed", error=str(exc))
        log.info("checkpointer.postgres_saver", dsn_host=s.postgres_dsn_host_only())
        return saver
    except Exception as exc:
        log.warning(
            "checkpointer.postgres_unavailable_fallback_to_memory",
            error=str(exc),
        )
        return MemorySaver()
