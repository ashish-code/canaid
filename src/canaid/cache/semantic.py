"""Turn-level Redis cache with intent-aware write policy."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog

from canaid.config import get_settings

log = structlog.get_logger(__name__)

# Intents whose answers depend only on the input (not session state).
# Anything outside this set is *not* cached on the write path.
_CACHEABLE_INTENTS: frozenset[str] = frozenset({
    "catalog_question",
    "small_talk",
})


def _normalize(message: str) -> str:
    return " ".join(message.lower().split())


def _key(message: str) -> str:
    digest = hashlib.sha256(_normalize(message).encode("utf-8")).hexdigest()
    return f"canaid:turn:v1:{digest}"


class TurnCache:
    """Stores a list of SSE frames keyed by the normalized user message."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = self._connect()

    def _connect(self):
        try:
            import redis  # local import — keeps cold-start light when disabled
            return redis.Redis.from_url(self._settings.redis_url, decode_responses=True)
        except Exception as exc:
            log.info("cache.redis_unavailable", error=str(exc))
            return None

    @property
    def enabled(self) -> bool:
        return bool(self._settings.cache_enabled and self._redis is not None)

    # ---- read ------------------------------------------------------
    def get(self, message: str) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        try:
            raw = self._redis.get(_key(message))  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("cache.read_failed", error=str(exc))
            return None
        if not raw:
            return None
        try:
            frames = json.loads(raw)
        except json.JSONDecodeError:
            return None
        log.info("cache.hit", key=_key(message), frames=len(frames))
        return frames

    # ---- write -----------------------------------------------------
    def set_if_cacheable(
        self,
        message: str,
        frames: list[dict[str, Any]],
        *,
        intent: str | None,
    ) -> bool:
        if not self.enabled or not frames:
            return False
        if intent not in _CACHEABLE_INTENTS:
            log.info("cache.skip", reason="non_cacheable_intent", intent=intent)
            return False
        try:
            self._redis.setex(  # type: ignore[union-attr]
                _key(message),
                self._settings.cache_ttl_seconds,
                json.dumps(frames),
            )
        except Exception as exc:
            log.warning("cache.write_failed", error=str(exc))
            return False
        log.info(
            "cache.set",
            key=_key(message),
            ttl=self._settings.cache_ttl_seconds,
            frames=len(frames),
            intent=intent,
        )
        return True


@lru_cache(maxsize=1)
def get_turn_cache() -> TurnCache:
    return TurnCache()
