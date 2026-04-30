"""Append-only audit log.

Records one row per turn with PII redacted. The shape is intentionally
flat so a SQL analyst can query it directly; nothing beyond `metadata`
is JSON.

Storage backend selection:
  * Postgres if available (`audit_events` table created by
    `scripts/sql/02-audit.sql`).
  * Falls through to a structured log line if Postgres is unreachable —
    "audit always succeeds" is the contract; degrading to logs preserves
    that.

Phase 9 will swap Postgres for a DynamoDB-on-demand table (cheaper for
write-mostly access patterns + native TTL for retention policy).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from canaid.config import get_settings
from canaid.guardrails.log_filter import regex_redact

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class AuditEvent:
    request_id: str
    conversation_id: str
    user_message_redacted: str
    intent: str | None
    user_type: str | None
    confidence: float | None
    route: str | None
    response_redacted: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditWriter:
    """Persist audit rows to Postgres, or log if Postgres is unavailable."""

    _INSERT_SQL = """
        INSERT INTO audit_events (
          request_id, conversation_id, user_message_redacted, intent, user_type,
          confidence, route, response_redacted,
          input_tokens, output_tokens, cost_usd, latency_ms, error, metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
    """

    def __init__(self) -> None:
        self._dsn = get_settings().pg_dsn
        self._available: bool | None = None

    def _conn(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def write(self, ev: AuditEvent) -> None:
        # Defensive PII scrub at the audit boundary — even if Bedrock
        # Guardrails has anonymized the LLM-bound copy, the *user input*
        # we received hasn't been touched yet.
        ev.user_message_redacted = regex_redact(ev.user_message_redacted or "")
        if ev.response_redacted:
            ev.response_redacted = regex_redact(ev.response_redacted)

        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    self._INSERT_SQL,
                    (
                        ev.request_id,
                        ev.conversation_id,
                        ev.user_message_redacted,
                        ev.intent,
                        ev.user_type,
                        ev.confidence,
                        ev.route,
                        ev.response_redacted,
                        ev.input_tokens,
                        ev.output_tokens,
                        ev.cost_usd,
                        ev.latency_ms,
                        ev.error,
                        json.dumps(ev.metadata),
                    ),
                )
            self._available = True
        except Exception as exc:
            self._available = False
            # Log path is the fallback. We log the row but mark the source so
            # an operator can backfill if needed.
            log.warning(
                "audit.persist_failed",
                error=str(exc),
                event=asdict(ev),
            )


_singleton: AuditWriter | None = None


def get_audit_writer() -> AuditWriter:
    global _singleton
    if _singleton is None:
        _singleton = AuditWriter()
    return _singleton
