"""Retriever — embeds a query and pulls top-k chunks.

Thin facade over `TitanEmbedder` + `PgVectorStore`. We keep it as its own
class for two reasons:
  1. It is the unit RAG agents depend on. Mocking the retriever in tests
     (return a fixed list of `RetrievedChunk`) is cleaner than mocking the
     store + embedder pair.
  2. Phase 9 will swap the store for OpenSearch Serverless. Agents won't
     change — they'll keep importing `get_retriever()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import structlog

from canaid.retrieval.embeddings import get_embedder
from canaid.retrieval.store import Hit, get_store

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    doc_type: str
    title: str
    content: str
    similarity: float
    metadata: dict[str, Any]

    @classmethod
    def from_hit(cls, hit: Hit) -> RetrievedChunk:
        return cls(
            doc_id=hit.doc_id,
            doc_type=hit.doc_type,
            title=hit.title,
            content=hit.content,
            similarity=hit.similarity,
            metadata=hit.metadata,
        )


class Retriever:
    def __init__(self, k: int = 5, min_similarity: float = 0.25) -> None:
        self.k = k
        self.min_similarity = min_similarity
        self._embedder = get_embedder()
        self._store = get_store()

    def search(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        vec = self._embedder.embed_one(query)
        hits = self._store.search(vec, k=k or self.k, doc_type=doc_type)
        kept = [
            RetrievedChunk.from_hit(h)
            for h in hits
            if h.similarity >= self.min_similarity
        ]
        log.info(
            "retriever.search",
            query_chars=len(query),
            hits_total=len(hits),
            hits_kept=len(kept),
            top_similarity=hits[0].similarity if hits else None,
        )
        return kept


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
