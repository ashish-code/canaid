"""Observability layer — logging, tracing, audit, LangFuse, cost."""

from canaid.observability.audit import AuditEvent, get_audit_writer
from canaid.observability.cost import cost_for, known_models
from canaid.observability.lf import get_langfuse_handler
from canaid.observability.logging import bind_request_context, configure_logging, get_logger

__all__ = [
    "AuditEvent",
    "bind_request_context",
    "configure_logging",
    "cost_for",
    "get_audit_writer",
    "get_langfuse_handler",
    "get_logger",
    "known_models",
]
