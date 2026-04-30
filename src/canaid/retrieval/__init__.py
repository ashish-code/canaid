"""Retrieval pipeline — chunking, embeddings, vector store, retriever.

Phase 3: pgvector-backed catalog + policy index. Phase 9 swaps the store
for OpenSearch Serverless via the same `Retriever` interface.
"""

from canaid.retrieval.chunker import Chunk, chunk_catalog, chunk_policies
from canaid.retrieval.embeddings import TitanEmbedder, get_embedder
from canaid.retrieval.retriever import RetrievedChunk, Retriever, get_retriever
from canaid.retrieval.store import PgVectorStore, get_store

__all__ = [
    "Chunk",
    "PgVectorStore",
    "RetrievedChunk",
    "Retriever",
    "TitanEmbedder",
    "chunk_catalog",
    "chunk_policies",
    "get_embedder",
    "get_retriever",
    "get_store",
]
