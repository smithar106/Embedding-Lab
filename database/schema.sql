-- Embedding-Lab schema.
--
-- Design principle: ONE logical chunk -> THREE embedding representations.
--
-- The chunk text is stored exactly once (in `chunks`). Each model's vectors
-- live in its own table because pgvector requires a FIXED dimension for a
-- vector column (and for the HNSW index), and MiniLM is 384-dim while BGE and
-- E5 are 768-dim. Three dimension-specific tables is the simplest technically
-- correct design; a single `vector` column cannot hold both 384- and 768-dim
-- values with a usable index.
--
-- Library model: every document belongs to a COLLECTION (e.g. climate, energy)
-- and carries provenance (source_organization, source_url, license) so the
-- frontend can render the library without hardcoding anything.

CREATE EXTENSION IF NOT EXISTS vector;

-- A source document as ingested (before chunking).
CREATE TABLE IF NOT EXISTS documents (
    document_id         TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    summary             TEXT NOT NULL DEFAULT '',
    collection          TEXT NOT NULL,
    topics              TEXT[] NOT NULL DEFAULT '{}',
    source_organization TEXT NOT NULL,
    source_url          TEXT,
    publication_date    DATE,
    retrieved_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    license             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,       -- sha256 of `text`, for incremental upsert
    text                TEXT NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection);
CREATE INDEX IF NOT EXISTS idx_documents_organization ON documents (source_organization);

-- The chunked text. Stored ONCE — identical for every embedding model.
-- Collection/topics/organization are denormalized here so retrieval can filter
-- and render results without joining back to `documents`.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    document_id         TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    chunk_text          TEXT NOT NULL,
    title               TEXT NOT NULL,
    collection          TEXT NOT NULL,
    topics              TEXT[] NOT NULL DEFAULT '{}',
    source_organization TEXT NOT NULL,
    source_url          TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash        TEXT NOT NULL,       -- sha256 of chunk_text, for incremental upsert
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks (collection);

-- MiniLM -> 384 dimensions.
CREATE TABLE IF NOT EXISTS chunk_embeddings_minilm (
    chunk_id   TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- BGE -> 768 dimensions.
CREATE TABLE IF NOT EXISTS chunk_embeddings_bge (
    chunk_id   TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- E5 -> 768 dimensions.
CREATE TABLE IF NOT EXISTS chunk_embeddings_e5 (
    chunk_id   TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW indexes for fast cosine-similarity search on each model's space.
CREATE INDEX IF NOT EXISTS idx_minilm_hnsw ON chunk_embeddings_minilm USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_bge_hnsw     ON chunk_embeddings_bge     USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_e5_hnsw      ON chunk_embeddings_e5      USING hnsw (embedding vector_cosine_ops);
