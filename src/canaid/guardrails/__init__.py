"""Guardrails — defense in depth for LLM I/O.

Three independent layers:

  1. **Bedrock Guardrails** (`policy.py`) — policy enforced *inside* the
     Bedrock Converse call. Topic deny-list (clinical advice, pricing
     quotes), content filters (hate/violence/insults/prompt-attack), PII
     anonymize/block. Created once via `scripts/setup_guardrail.py`,
     wired into every LLM call by ID.
  2. **Comprehend PII** (`pii.py`) — explicit detection on user input for
     audit/data-minimization. Uses AWS Comprehend's `detect_pii_entities`.
     Slower than regex; we only run it on the audit-log path, not every
     log line.
  3. **Regex log scrubber** (`log_filter.py`) — cheap structlog processor
     that strips email, phone, credit-card patterns from *every* log
     value. Belt-and-suspenders for the cases where a string sneaks
     through unredacted.

Plus templated refusals (`refusals.py`) so the user sees consistent
language whether the refusal comes from policy or intent classification.

Why three layers and not just Bedrock Guardrails:
  * Bedrock Guardrails runs at the LLM boundary. It can't redact what we
    log to CloudWatch outside the call.
  * Comprehend gives us per-entity types and offsets, which Bedrock
    Guardrails does not expose to the application.
  * Regex covers the cheap-and-always-on path. If Comprehend is throttled
    or unavailable, logs still don't leak.
"""

from canaid.guardrails.log_filter import pii_log_processor, regex_redact
from canaid.guardrails.pii import (
    PiiDetector,
    PiiEntity,
    get_pii_detector,
    redact_with_entities,
)
from canaid.guardrails.policy import guardrail_config_for_converse
from canaid.guardrails.refusals import refusal_text

__all__ = [
    "PiiDetector",
    "PiiEntity",
    "get_pii_detector",
    "guardrail_config_for_converse",
    "pii_log_processor",
    "redact_with_entities",
    "refusal_text",
    "regex_redact",
]
