"""pgvector-backed chunk store.

Direct psycopg + pgvector — we don't use SQLAlchemy or LangChain's
`PGVector` wrapper. Two reasons:

  1. **One source of truth for the schema.** `scripts/sql/01-rag.sql` is
     the migration; ORMs would duplicate that information in Python.
  2. **Latency.** A direct prepared statement on a single connection is
     faster than the abstraction tax of an ORM. The query is the hot path
     of every catalog turn.

Phase 9 swaps this store for OpenSearch Serverless behind the same
`Retriever` API. The retriever, the chunker, and the embedder remain
unchanged — only the store is region-specific.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# psycopg + pgvector imports happen INSIDE the class methods so the
# embedded (Streamlit Cloud / FAISS) deploy doesn't need them installed.
# `Hit` and the module are still importable without them.
from canaid.config import get_settings
from canaid.retrieval.chunker import Chunk


@dataclass
class Hit:
    doc_id: str
    doc_type: str
    chunk_index: int
    title: str
    content: str
    metadata: dict[str, Any]
    similarity: float          # 1 - cosine_distance, in [0, 1]


class PgVectorStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or get_settings().pg_dsn

    @contextmanager
    def _conn(self):
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            yield conn

    # ---- writes -----------------------------------------------------
    def upsert(
        self,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Idempotent insert by (doc_id, chunk_index). Replaces on conflict."""
        chunks = list(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError("chunks/embeddings length mismatch")
        rows = [
            (
                c.doc_id,
                c.doc_type,
                c.chunk_index,
                c.title,
                c.content,
                json.dumps(c.metadata),
                e,
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO catalog_chunks
                  (doc_id, doc_type, chunk_index, title, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                  doc_type = EXCLUDED.doc_type,
                  title    = EXCLUDED.title,
                  content  = EXCLUDED.content,
                  metadata = EXCLUDED.metadata,
                  embedding = EXCLUDED.embedding,
                  created_at = now()
                """,
                rows,
            )
        return len(rows)

    def delete_all(self) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM catalog_chunks")

    def count(self) -> int:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM catalog_chunks")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # ---- reads ------------------------------------------------------
    def search(
        self,
        query_vec: list[float],
        k: int = 5,
        doc_type: str | None = None,
    ) -> list[Hit]:
        from psycopg.rows import dict_row

        sql = """
            SELECT doc_id, doc_type, chunk_index, title, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM catalog_chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        where = ""
        params: list[Any] = [query_vec]
        if doc_type:
            where = "WHERE doc_type = %s"
            params.append(doc_type)
        params.extend([query_vec, k])

        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql.format(where=where), params)
            return [
                Hit(
                    doc_id=row["doc_id"],
                    doc_type=row["doc_type"],
                    chunk_index=row["chunk_index"],
                    title=row["title"],
                    content=row["content"],
                    metadata=row["metadata"] or {},
                    similarity=float(row["similarity"]),
                )
                for row in cur.fetchall()
            ]


_singleton: PgVectorStore | None = None


def get_store() -> PgVectorStore:
    global _singleton
    if _singleton is None:
        _singleton = PgVectorStore()
    return _singleton
