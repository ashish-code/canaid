"""FAISS store unit tests — no AWS calls, no Bedrock embeddings.

We construct synthetic vectors directly so the index logic (insert,
search, doc_type filter, normalization) is testable offline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from canaid.retrieval.chunker import Chunk
from canaid.retrieval.faiss_store import FaissStore


def _vec(dim: int, *, dominant: int) -> list[float]:
    """Return a vector with `1.0` at `dominant`, small noise elsewhere."""
    v = [0.01] * dim
    v[dominant] = 1.0
    return v


def _chunk(doc_id: str, doc_type: str = "sku") -> Chunk:
    return Chunk(
        doc_id=doc_id,
        doc_type=doc_type,
        chunk_index=0,
        title=f"title-{doc_id}",
        content=f"content for {doc_id}",
        metadata={"doc_id": doc_id},
    )


def test_search_returns_nearest() -> None:
    store = FaissStore(dim=8)
    store.upsert(
        [_chunk("A"), _chunk("B"), _chunk("C")],
        [_vec(8, dominant=0), _vec(8, dominant=1), _vec(8, dominant=2)],
    )
    hits = store.search(_vec(8, dominant=1), k=2)
    assert hits[0].doc_id == "B"
    assert len(hits) == 2


def test_search_returns_similarity_scores() -> None:
    store = FaissStore(dim=4)
    store.upsert([_chunk("A")], [_vec(4, dominant=0)])
    hits = store.search(_vec(4, dominant=0), k=1)
    # Self-similarity over L2-normalized inner-product is ~1.0
    assert hits[0].similarity > 0.99


def test_doc_type_filter_excludes_other_types() -> None:
    store = FaissStore(dim=4)
    store.upsert(
        [_chunk("S1", "sku"), _chunk("P1", "policy"), _chunk("S2", "sku")],
        [_vec(4, dominant=0), _vec(4, dominant=0), _vec(4, dominant=1)],
    )
    hits = store.search(_vec(4, dominant=0), k=10, doc_type="policy")
    assert all(h.doc_type == "policy" for h in hits)
    assert {h.doc_id for h in hits} == {"P1"}


def test_empty_store_returns_no_hits() -> None:
    store = FaissStore(dim=4)
    assert store.search(_vec(4, dominant=0), k=5) == []


def test_count_tracks_upserts() -> None:
    store = FaissStore(dim=4)
    assert store.count() == 0
    store.upsert([_chunk("A"), _chunk("B")], [_vec(4, dominant=0), _vec(4, dominant=1)])
    assert store.count() == 2


def test_delete_all_clears_index() -> None:
    store = FaissStore(dim=4)
    store.upsert([_chunk("A")], [_vec(4, dominant=0)])
    store.delete_all()
    assert store.count() == 0
    assert store.search(_vec(4, dominant=0), k=5) == []


def test_length_mismatch_raises() -> None:
    store = FaissStore(dim=4)
    with pytest.raises(ValueError):
        store.upsert([_chunk("A"), _chunk("B")], [_vec(4, dominant=0)])
