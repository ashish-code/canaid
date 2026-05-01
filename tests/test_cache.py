"""TurnCache tests with a fake backend.

We don't talk to a real Redis here; the tests verify the cache's *logic*:
key normalization, intent gating, graceful degradation. The in-memory
backend has its own integration test below.
"""

from __future__ import annotations

import time

from canaid.cache.semantic import TurnCache, _key, _MemoryBackend, _normalize


class _FakeBackend:
    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail = fail

    def get(self, key: str):
        if self.fail:
            raise RuntimeError("simulated backend down")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        if self.fail:
            raise RuntimeError("simulated backend down")
        self.store[key] = value


def _build(fake) -> TurnCache:
    cache = TurnCache.__new__(TurnCache)
    from canaid.config import get_settings

    cache._settings = get_settings()
    cache._backend = fake
    cache._kind = "fake"
    return cache


def test_key_normalizes_whitespace_and_case() -> None:
    a = _key("  Hello World  ")
    b = _key("hello world")
    assert a == b


def test_key_includes_version_prefix() -> None:
    assert _key("anything").startswith("canaid:turn:v1:")


def test_normalize_collapses_multispace() -> None:
    assert _normalize("a   b\nc") == "a b c"


def test_get_returns_none_when_missing() -> None:
    cache = _build(_FakeBackend())
    assert cache.get("not in cache") is None


def test_set_and_get_roundtrip_for_cacheable_intent() -> None:
    cache = _build(_FakeBackend())
    frames = [{"type": "token", "data": "hi"}]
    wrote = cache.set_if_cacheable("hello", frames, intent="catalog_question")
    assert wrote is True
    assert cache.get("hello") == frames


def test_set_skips_non_cacheable_intent() -> None:
    cache = _build(_FakeBackend())
    wrote = cache.set_if_cacheable(
        "lookup my account", [{"type": "token", "data": "x"}], intent="account_lookup"
    )
    assert wrote is False
    assert cache.get("lookup my account") is None


def test_set_skips_when_intent_unknown() -> None:
    cache = _build(_FakeBackend())
    wrote = cache.set_if_cacheable("?", [{"type": "token", "data": "x"}], intent=None)
    assert wrote is False


def test_backend_failures_degrade_gracefully() -> None:
    cache = _build(_FakeBackend(fail=True))
    # Both reads and writes silently no-op rather than raising.
    assert cache.get("anything") is None
    assert (
        cache.set_if_cacheable("q", [{"type": "token", "data": "x"}], intent="catalog_question")
        is False
    )


def test_disabled_cache_is_no_op() -> None:
    cache = TurnCache.__new__(TurnCache)
    from canaid.config import Settings

    cache._settings = Settings(cache_enabled=False)
    cache._backend = _FakeBackend()
    cache._kind = "fake"
    assert cache.enabled is False
    assert cache.get("hello") is None
    assert cache.set_if_cacheable("hello", [{"type": "token", "data": "x"}], intent="catalog_question") is False


# ---- in-memory backend ---------------------------------------------------
def test_memory_backend_set_then_get() -> None:
    backend = _MemoryBackend()
    backend.setex("k1", ttl=60, value="hello")
    assert backend.get("k1") == "hello"


def test_memory_backend_expires_after_ttl() -> None:
    backend = _MemoryBackend()
    backend.setex("k2", ttl=0, value="hello")
    # Sleep just enough for monotonic time to advance past ttl=0
    time.sleep(0.001)
    assert backend.get("k2") is None


def test_memory_backend_returns_none_for_missing_key() -> None:
    backend = _MemoryBackend()
    assert backend.get("never-set") is None
