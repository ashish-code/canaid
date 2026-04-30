"""TurnCache tests with a fake Redis client.

We don't talk to a real Redis here — that's covered by the integration
tests when ``make infra-up`` is running. These tests verify the cache's
*logic*: key normalization, intent gating, graceful degradation.
"""

from __future__ import annotations

from canaid.cache.semantic import TurnCache, _key, _normalize


class _FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail = fail

    def get(self, key: str):
        if self.fail:
            raise RuntimeError("simulated redis down")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        if self.fail:
            raise RuntimeError("simulated redis down")
        self.store[key] = value


def _build(fake) -> TurnCache:
    cache = TurnCache.__new__(TurnCache)
    from canaid.config import get_settings

    cache._settings = get_settings()
    cache._redis = fake
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
    cache = _build(_FakeRedis())
    assert cache.get("not in cache") is None


def test_set_and_get_roundtrip_for_cacheable_intent() -> None:
    cache = _build(_FakeRedis())
    frames = [{"type": "token", "data": "hi"}]
    wrote = cache.set_if_cacheable("hello", frames, intent="catalog_question")
    assert wrote is True
    assert cache.get("hello") == frames


def test_set_skips_non_cacheable_intent() -> None:
    cache = _build(_FakeRedis())
    wrote = cache.set_if_cacheable(
        "lookup my account", [{"type": "token", "data": "x"}], intent="account_lookup"
    )
    assert wrote is False
    assert cache.get("lookup my account") is None


def test_set_skips_when_intent_unknown() -> None:
    cache = _build(_FakeRedis())
    wrote = cache.set_if_cacheable("?", [{"type": "token", "data": "x"}], intent=None)
    assert wrote is False


def test_redis_failures_degrade_gracefully() -> None:
    cache = _build(_FakeRedis(fail=True))
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
    cache._redis = _FakeRedis()
    assert cache.enabled is False
    assert cache.get("hello") is None
    assert cache.set_if_cacheable("hello", [{"type": "token", "data": "x"}], intent="catalog_question") is False
