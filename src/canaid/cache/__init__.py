"""Response caches.

Phase 6 ships a single layer: a turn-level Redis cache keyed by the
normalized user message. On hit, the API replays the cached SSE frames
and skips the LangGraph invocation entirely — saving the embed call,
the retrieve, and the LLM tokens. We only cache intents that are
deterministic given the input (catalog questions, small talk); other
intents (lookup, qualifier) carry session-specific state that would
poison the cache.

Why exact-match (not semantic) for now:
  * Simple: no extra Redis modules (RedisSearch / RediSearch) needed.
  * Cheap: ``GET <hash(query)>`` is sub-millisecond.
  * Correctness over cleverness: an LSH or k-NN cache that returns a
    near-duplicate's answer is plausible-sounding *failure* mode. Demo
    audiences don't trust caches that hallucinate by analogy.

The path to semantic match is documented in ``docs/06-caching-memory.md``
— LSH bucket key from the embedding's sign bits is the typical next step
when exact-match hit rate plateaus.
"""

from canaid.cache.semantic import TurnCache, get_turn_cache

__all__ = ["TurnCache", "get_turn_cache"]
