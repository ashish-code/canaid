"""Chunker unit tests — no AWS, no Postgres."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canaid.retrieval.chunker import chunk_catalog, chunk_policies

REPO = Path(__file__).resolve().parents[1]


def test_catalog_emits_one_chunk_per_sku() -> None:
    skus = [
        json.loads(line)
        for line in (REPO / "data/catalog/skus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    chunks = list(chunk_catalog(REPO / "data/catalog/skus.jsonl"))
    assert len(chunks) == len(skus)
    assert all(c.doc_type == "sku" for c in chunks)
    assert all(c.chunk_index == 0 for c in chunks)


def test_catalog_chunk_text_is_human_readable() -> None:
    chunks = list(chunk_catalog(REPO / "data/catalog/skus.jsonl"))
    sample = next(c for c in chunks if c.doc_id == "GLOVE-NTR-M-100")
    text = sample.content
    assert sample.title in text or "Nitrile" in text
    assert "SKU:" in text and "GLOVE-NTR-M-100" in text
    assert "Specs" in text
    # JSON must be flattened — no raw JSON braces in chunk content.
    assert "{" not in text
    assert "}" not in text


def test_catalog_metadata_preserves_filterable_fields() -> None:
    chunks = list(chunk_catalog(REPO / "data/catalog/skus.jsonl"))
    sample = next(c for c in chunks if c.doc_id == "ANTISEPT-CHX-500")
    assert sample.metadata["category"] == "WoundCare"
    assert sample.metadata["country_of_origin"] == "Canada"


def test_policies_chunked() -> None:
    chunks = list(chunk_policies(REPO / "data/policies"))
    doc_ids = {c.doc_id for c in chunks}
    assert {"shipping", "returns", "ordering", "compliance"} <= doc_ids
    # Each policy file should produce >= 1 chunk.
    assert len(chunks) >= 4


def test_policy_chunks_have_indexes_and_titles() -> None:
    chunks = [c for c in chunk_policies(REPO / "data/policies") if c.doc_id == "shipping"]
    indexes = [c.chunk_index for c in chunks]
    assert indexes == sorted(indexes)
    assert indexes[0] == 0
    assert all(c.title for c in chunks)


def test_unknown_sku_path_raises_quickly() -> None:
    with pytest.raises(FileNotFoundError):
        list(chunk_catalog(Path("/no/such/path.jsonl")))
