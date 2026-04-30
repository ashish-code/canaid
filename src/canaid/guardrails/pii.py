"""AWS Comprehend PII detection wrapper.

Used for the *audit* and *data-minimization* paths: when we persist a
user-turn to S3 / DynamoDB for replay or eval, we redact through this
module so the stored copy never carries unredacted PII.

Why Comprehend (vs. regex-only):
  * Catches name and address spans that no regex will get reliably.
  * Returns confidence scores per entity, useful for a tunable threshold.
  * AWS-native — no extra vendor onboarding.

Why *not* Comprehend on every log line:
  * Comprehend is a network call ($, latency). Logs are emitted from hot
    paths. The regex scrubber in `log_filter.py` covers the easy cases on
    every line; Comprehend covers the audit/eval persistence path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import structlog

from canaid.config import boto_client_factory_safe

log = structlog.get_logger(__name__)

# Entity types we treat as "definitely redact" from the audit log.
_HARD_REDACT_TYPES: frozenset[str] = frozenset({
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "USERNAME",
    "PASSWORD",
    "CREDIT_DEBIT_NUMBER",
    "CREDIT_DEBIT_CVV",
    "CREDIT_DEBIT_EXPIRY",
    "PIN",
    "BANK_ACCOUNT_NUMBER",
    "BANK_ROUTING",
    "SSN",
    "PASSPORT_NUMBER",
    "DRIVER_ID",
    "LICENSE_PLATE",
    "DATE_TIME",
    "AGE",
    "NAME",
    "URL",
    "IP_ADDRESS",
    "MAC_ADDRESS",
})


@dataclass(frozen=True, slots=True)
class PiiEntity:
    type: str
    start: int
    end: int
    score: float


class PiiDetector:
    """Wrapper around `comprehend.detect_pii_entities`."""

    def __init__(self, language_code: str = "en") -> None:
        self.language_code = language_code
        self._client = boto_client_factory_safe("comprehend")

    def detect(self, text: str) -> list[PiiEntity]:
        if not text or len(text.strip()) < 3 or self._client is None:
            return []
        try:
            resp = self._client.detect_pii_entities(
                Text=text, LanguageCode=self.language_code
            )
        except Exception as exc:  # network / throttling / IAM
            log.warning("pii.detect_failed", error=str(exc))
            return []
        return [
            PiiEntity(
                type=e["Type"],
                start=int(e["BeginOffset"]),
                end=int(e["EndOffset"]),
                score=float(e.get("Score", 0.0)),
            )
            for e in resp.get("Entities", [])
        ]


def redact_with_entities(
    text: str,
    entities: list[PiiEntity],
    *,
    types_to_redact: frozenset[str] | None = None,
    placeholder_fmt: str = "[{type}]",
) -> str:
    """Apply redactions to `text` based on detected entities.

    Walks entities in reverse offset order so substitutions never invalidate
    later offsets.
    """
    keep_types = types_to_redact or _HARD_REDACT_TYPES
    out = text
    for e in sorted(entities, key=lambda x: x.start, reverse=True):
        if e.type not in keep_types:
            continue
        out = out[: e.start] + placeholder_fmt.format(type=e.type) + out[e.end :]
    return out


@lru_cache(maxsize=1)
def get_pii_detector() -> PiiDetector:
    return PiiDetector()
