"""Build (or rebuild) the RAG index from the local data files.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --reset    # delete first

Reads:
    data/catalog/skus.jsonl
    data/policies/*.md

Writes to the `catalog_chunks` table in Postgres (DSN from CANAID_PG_DSN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from canaid.observability.logging import configure_logging, get_logger
from canaid.retrieval.chunker import chunk_catalog, chunk_policies
from canaid.retrieval.embeddings import get_embedder
from canaid.retrieval.store import get_store

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "data" / "catalog" / "skus.jsonl"
POLICIES = REPO_ROOT / "data" / "policies"


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="delete all rows from catalog_chunks before ingesting",
    )
    args = parser.parse_args()

    store = get_store()
    embedder = get_embedder()

    if args.reset:
        store.delete_all()
        log.info("build_index.reset")

    chunks = list(chunk_catalog(CATALOG)) + list(chunk_policies(POLICIES))
    log.info("build_index.chunks_collected", n=len(chunks))

    if not chunks:
        log.error("build_index.no_chunks", catalog=str(CATALOG), policies=str(POLICIES))
        return 1

    embeddings = embedder.embed_many(c.content for c in chunks)
    log.info("build_index.embeddings_done", n=len(embeddings))

    written = store.upsert(chunks, embeddings)
    log.info("build_index.upserted", n=written, total=store.count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
