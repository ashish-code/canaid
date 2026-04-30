"""Turn-level cache with intent-aware write policy.

Two backends:

  * **Redis** — preferred. Survives process restarts, shareable across
    replicas, native TTL.
  * **In-memory dict with TTL** — fallback when Redis isn't reachable
    (e.g., Streamlit Community Cloud). Per-process; loses state on
    restart but still saves the LLM call within a session.

The TurnCache class hides the choice — same API both ways.
"""

from __future__ import annotations

import hashlib
import json
import time
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


class _MemoryBackend:
    """Per-process dict with monotonic-time TTL. Tiny, no deps."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (time.monotonic() + ttl, value)

    def __len__(self) -> int:
        return len(self._store)


class TurnCache:
    """Stores a list of SSE frames keyed by the normalized user message."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._backend, self._kind = self._connect()

    def _connect(self) -> tuple[Any, str]:
        # In embedded mode (Streamlit Cloud) we don't even attempt Redis —
        # avoids a noisy connect-fail on every cold start.
        if self._settings.embedded:
            return _MemoryBackend(), "memory"
        try:
            import redis  # local import — keeps cold-start light when disabled
            client = redis.Redis.from_url(self._settings.redis_url, decode_responses=True)
            # Cheap liveness probe — fails fast if Redis is unreachable.
            client.ping()
            return client, "redis"
        except Exception as exc:
            log.info("cache.redis_unavailable_using_memory", error=str(exc))
            return _MemoryBackend(), "memory"

    @property
    def enabled(self) -> bool:
        return bool(self._settings.cache_enabled and self._backend is not None)

    @property
    def kind(self) -> str:
        return self._kind

    # ---- read ------------------------------------------------------
    def get(self, message: str) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        try:
            raw = self._backend.get(_key(message))
        except Exception as exc:
            log.warning("cache.read_failed", error=str(exc))
            return None
        if not raw:
            return None
        try:
            frames = json.loads(raw)
        except json.JSONDecodeError:
            return None
        log.info("cache.hit", key=_key(message), frames=len(frames), kind=self._kind)
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
            self._backend.setex(
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
            kind=self._kind,
        )
        return True


@lru_cache(maxsize=1)
def get_turn_cache() -> TurnCache:
    return TurnCache()
