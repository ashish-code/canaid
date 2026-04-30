"""Cheap regex-based PII scrubber for logs.

Runs as a structlog processor on every log line. Catches the patterns that
*can* be matched by regex (email, phone, credit-card, common ID formats).
For free-form name/address spans we rely on the Comprehend layer in the
audit/eval paths.

Idempotent — running it twice produces the same output.
"""

from __future__ import annotations

import re
from typing import Any

# Tightened a bit from the typical "any 16 digits" — must include separators
# or be flanked by word-boundaries to avoid eating order numbers.
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Country-code prefix is optional (most matches in our domain don't carry one).
_RE_PHONE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
)
_RE_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
_RE_SIN = re.compile(r"\b\d{3}[ \-]?\d{3}[ \-]?\d{3}\b")  # Canadian SIN format

_REPLACEMENTS = (
    (_RE_EMAIL, "[EMAIL]"),
    (_RE_CARD, "[CARD]"),  # before phone — card matches a longer span
    (_RE_PHONE, "[PHONE]"),
    (_RE_SIN, "[SIN]"),
)


def regex_redact(text: str) -> str:
    out = text
    for pattern, repl in _REPLACEMENTS:
        out = pattern.sub(repl, out)
    return out


# Keys that are obviously non-PII — skip the regex on these for speed.
_SAFE_KEYS = frozenset({
    "level", "timestamp", "logger", "event", "request_id", "conversation_id",
    "model_id", "agent", "vendor", "cost_tier", "input_tokens", "output_tokens",
    "latency_ms", "stop_reason", "tool_calls", "hits", "n", "processed",
})


def pii_log_processor(_logger, _method_name, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor — redacts string values in event_dict in place."""
    for k, v in list(event_dict.items()):
        if k in _SAFE_KEYS:
            continue
        if isinstance(v, str) and len(v) > 4:
            scrubbed = regex_redact(v)
            if scrubbed != v:
                event_dict[k] = scrubbed
    return event_dict
