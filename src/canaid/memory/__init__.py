"""Conversation memory.

Phase 6 ships the LangGraph checkpointer factory only:
  * Default: ``MemorySaver`` — fine for local dev; non-durable.
  * Production: ``PostgresSaver`` — durable across container restarts and
    rolling deploys, enabled via ``CANAID_USE_POSTGRES_CHECKPOINTER=true``.

Phase 8 adds an audit-log writer (DynamoDB) and a long-term user-profile
store (Postgres ``user_profiles``). Those concerns are intentionally
*separate* from the checkpointer:

  * **Checkpointer** — short-term graph state, scoped to ``thread_id``,
    used by LangGraph internally to resume conversations across crashes.
  * **Audit log** — append-only, immutable, PII-redacted, retained per
    policy. Not the same as graph state.
  * **User profile** — long-term per-user/per-account preferences,
    queryable, joinable. Different schema, different lifetime.
"""

from canaid.memory.checkpointer import get_checkpointer

__all__ = ["get_checkpointer"]
