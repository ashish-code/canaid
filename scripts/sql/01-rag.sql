-- RAG schema for the catalog + policies index.
--
-- One table; rows are individual chunks. The HNSW index uses cosine
-- distance because Titan Embed v2 vectors are L2-normalized — cosine
-- gives stable similarity scores across query and document spaces.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid

CREATE TABLE IF NOT EXISTS catalog_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id        TEXT NOT NULL,                         -- SKU id or policy filename
    doc_type      TEXT NOT NULL CHECK (doc_type IN ('sku','policy')),
    chunk_index   INT  NOT NULL,
    title         TEXT,
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS catalog_chunks_doc_idx
    ON catalog_chunks (doc_id);
CREATE INDEX IF NOT EXISTS catalog_chunks_type_idx
    ON catalog_chunks (doc_type);

-- HNSW with cosine distance. m=16, ef_construction=64 are pgvector defaults
-- and adequate for a few hundred rows; we'll tune in Phase 9 once the index
-- crosses ~10k rows.
CREATE INDEX IF NOT EXISTS catalog_chunks_embedding_hnsw
    ON catalog_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
