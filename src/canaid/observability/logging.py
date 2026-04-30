"""Structured logging via structlog.

Why structlog: every log line is a single JSON record (`ts`, `level`, `event`,
plus context). Lets us grep CloudWatch / Loki / Datadog by structured field
instead of regexing free text. We bind context via contextvars so that
log lines emitted deep inside an agent automatically carry the
`request_id` / `conversation_id` / `agent` of the surrounding request.

PII redaction will plug in here as a structlog processor in Phase 5 — that's
the right place because every log line passes through processors before
serialization, so we never leak even if a downstream caller forgets to
sanitize.
"""

from __future__ import annotations

import logging
import sys

import structlog

from canaid.config import get_settings


def configure_logging() -> None:
    """Idempotent. Call once at app startup."""
    if getattr(configure_logging, "_done", False):
        return

    level = getattr(logging, get_settings().log_level)

    # Local import to avoid an import cycle: guardrails depends on config,
    # config doesn't depend on guardrails.
    from canaid.guardrails.log_filter import pii_log_processor

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # PII scrub runs LAST among non-renderer processors so it sees the
        # final shape of every event_dict.
        pii_log_processor,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Quiet noisy libraries.
    for noisy in ("boto3", "botocore", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    configure_logging._done = True  # type: ignore[attr-defined]


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


def bind_request_context(**kwargs) -> None:
    """Bind contextvars that all subsequent log lines inherit until cleared."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
