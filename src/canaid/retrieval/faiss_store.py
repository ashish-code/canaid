"""In-memory FAISS-backed chunk store.

Used in the **single-process** deployment (Streamlit Community Cloud)
where pgvector isn't available. Same `Hit` contract as `PgVectorStore`,
so the `Retriever` facade swaps stores transparently.

Index is built once at startup from `data/catalog/skus.jsonl` +
`data/policies/*.md`. Embedding goes through Bedrock Titan v2 (one call
per chunk, ~25 calls — sub-15-second cold start). The index is held in
memory; a second process means a second index, which is fine at this
scale.

We use IndexFlatIP (inner-product over L2-normalized vectors = cosine).
For ~25 chunks IP is plenty; HNSW would be over-engineering. The
identical similarity scores let us reuse the Phase 3 `min_similarity`
threshold without retuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from canaid.retrieval.chunker import Chunk, chunk_catalog, chunk_policies
from canaid.retrieval.store import Hit

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "data" / "catalog" / "skus.jsonl"
POLICIES_DIR = REPO_ROOT / "data" / "policies"


@dataclass(slots=True)
class _IndexedChunk:
    chunk: Chunk
    vector_idx: int     # row in the FAISS index


class FaissStore:
    """Drop-in replacement for `PgVectorStore` for embedded deploys."""

    def __init__(self, dim: int = 1024) -> None:
        # Defer faiss + numpy imports until construction so the package
        # stays optional for non-embedded deploys.
        import faiss  # type: ignore[import-not-found]
        import numpy as np

        self._faiss = faiss
        self._np = np
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    # ---- writes -----------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks/embeddings length mismatch")
        if not chunks:
            return 0
        arr = self._np.asarray(embeddings, dtype="float32")
        # Titan v2 returns L2-normalized vectors, but normalize again
        # defensively — a single non-normalized vector skews IP scores.
        self._faiss.normalize_L2(arr)
        self._index.add(arr)
        self._chunks.extend(chunks)
        log.info("faiss.upsert", n=len(chunks), total=len(self._chunks))
        return len(chunks)

    def delete_all(self) -> None:
        self._index.reset()
        self._chunks.clear()

    def count(self) -> int:
        return len(self._chunks)

    # ---- reads ------------------------------------------------------
    def search(
        self,
        query_vec: list[float],
        k: int = 5,
        doc_type: str | None = None,
    ) -> list[Hit]:
        if not self._chunks:
            return []
        q = self._np.asarray([query_vec], dtype="float32")
        self._faiss.normalize_L2(q)
        # Over-fetch when filtering by doc_type so we still hit k after
        # the filter pass.
        fetch_k = k * 4 if doc_type else k
        scores, indices = self._index.search(q, min(fetch_k, len(self._chunks)))

        hits: list[Hit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if doc_type and chunk.doc_type != doc_type:
                continue
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    doc_type=chunk.doc_type,
                    chunk_index=chunk.chunk_index,
                    title=chunk.title or "",
                    content=chunk.content,
                    metadata=dict(chunk.metadata or {}),
                    similarity=float(score),
                )
            )
            if len(hits) >= k:
                break
        return hits


def build_faiss_store_from_data() -> FaissStore:
    """Build a FaissStore from the local data files. Embeds via Bedrock."""
    from canaid.retrieval.embeddings import EMBED_DIM, get_embedder

    log.info("faiss.build_start", catalog=str(CATALOG_PATH), policies=str(POLICIES_DIR))
    chunks: list[Chunk] = []
    if CATALOG_PATH.exists():
        chunks.extend(chunk_catalog(CATALOG_PATH))
    if POLICIES_DIR.exists():
        chunks.extend(chunk_policies(POLICIES_DIR))
    if not chunks:
        log.warning("faiss.no_data")
        return FaissStore(dim=EMBED_DIM)

    embedder = get_embedder()
    vectors = embedder.embed_many(c.content for c in chunks)
    store = FaissStore(dim=EMBED_DIM)
    store.upsert(chunks, vectors)
    log.info("faiss.build_done", n=store.count())
    return store


# Module-level cached instance — Streamlit reruns the app on every input,
# so we MUST not rebuild the index per turn. Streamlit's `@st.cache_resource`
# on the wrapper layer does the same job from the UI side; this lru_cache
# protects any non-Streamlit caller.
_singleton: FaissStore | None = None


def get_faiss_store() -> FaissStore:
    global _singleton
    if _singleton is None:
        _singleton = build_faiss_store_from_data()
    return _singleton
