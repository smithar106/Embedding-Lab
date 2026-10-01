"""The indexing pipeline: load -> chunk ONCE -> embed THREE times -> store.

INDEXING TIME is when the heavy, one-off work happens. For every chunk we run
the three embedding models and persist the results, so that QUERY TIME can be
fast: a query is embedded once and looked up against already-stored vectors.

    documents -> chunks -> document embeddings -> pgvector

The pipeline is incremental: each chunk carries a ``content_hash``, so if you
re-run ingestion on an unchanged document, the unchanged chunks are NOT
re-embedded — only new or modified chunks are. This is the hook that will later
let a real dataset refresh in place without re-embedding everything.
"""
from __future__ import annotations

import time

from psycopg.types.json import Jsonb

from database import MODEL_TABLES
from database.connection import get_connection, init_db, reset_db
from ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document
from ingestion.loader import load_documents
from models.embedding_models import MODELS, embed_document


def index_dataset(
    path: str,
    *,
    models: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reset: bool = False,
    verbose: bool = True,
) -> dict:
    """Index a dataset: load -> chunk once -> embed with every model -> store."""
    models = models or list(MODELS)
    started = time.perf_counter()

    conn = get_connection()
    try:
        init_db(conn)
        if reset:
            reset_db(conn)

        # ---- 1. Load + normalize -------------------------------------------
        documents = load_documents(path)
        if verbose:
            print(f"Loaded: {len(documents)} documents")

        # ---- 2. Chunk ONCE (identical chunks for all models) ---------------
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        if verbose:
            print(f"Created: {len(all_chunks):,} chunks")

        # ---- 3. Upsert documents ------------------------------------------
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    """
                    INSERT INTO documents (document_id, title, source, original_text, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        original_text = EXCLUDED.original_text,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    """,
                    (doc["document_id"], doc["title"], doc["source"], doc["text"], _json(doc["metadata"])),
                )
        conn.commit()

        # ---- 4. Decide which chunks are new/changed vs unchanged ----------
        existing_hashes = {}
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, content_hash FROM chunks")
            for chunk_id, content_hash in cur.fetchall():
                existing_hashes[chunk_id] = content_hash

        changed: list[dict] = []
        unchanged_count = 0
        for c in all_chunks:
            if existing_hashes.get(c["chunk_id"]) == c["content_hash"]:
                unchanged_count += 1
            else:
                changed.append(c)

        # ---- 5. Upsert chunks (text stored once) ---------------------------
        with conn.cursor() as cur:
            for c in all_chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, document_id, chunk_index, chunk_text, title, source, metadata, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        content_hash = EXCLUDED.content_hash,
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata
                    """,
                    (c["chunk_id"], c["document_id"], c["chunk_index"], c["chunk_text"],
                     c["title"], c["source"], _json(c["metadata"]), c["content_hash"]),
                )

        # Remove chunks that no longer exist after re-chunking (e.g. a document
        # shrank). The FK cascade removes their embeddings too.
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    "DELETE FROM chunks WHERE document_id = %s AND chunk_index >= %s",
                    (doc["document_id"], len(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))),
                )
        conn.commit()

        # ---- 6. Embed ONLY new/changed chunks, once per model --------------
        if verbose and changed:
            print(f"\nEmbedding {len(changed):,} changed chunk(s) with each model…")
        embeddings_generated = 0
        for model_name in models:
            table = MODEL_TABLES[model_name]
            if verbose:
                print(f"\nEmbedding {model_name.upper()}…")
            with conn.cursor() as cur:
                for i, c in enumerate(changed, start=1):
                    vector = embed_document(c["chunk_text"], model_name)
                    cur.execute(
                        f"""
                        INSERT INTO {table} (chunk_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now()
                        """,
                        (c["chunk_id"], vector),
                    )
                    embeddings_generated += 1
                    if verbose and (i % 100 == 0 or i == len(changed)):
                        print(f"  {i:,} / {len(changed):,}")
            conn.commit()

        elapsed = time.perf_counter() - started
        if verbose:
            print("\nIndexing complete.")

        return {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "chunks_unchanged": unchanged_count,
            "chunks_changed": len(changed),
            "embeddings_generated": embeddings_generated,
            "embeddings_skipped": unchanged_count * len(models),
            "runtime_seconds": round(elapsed, 2),
        }
    finally:
        conn.close()


def _json(metadata: dict):
    """Wrap a dict for psycopg3 so it adapts to a PostgreSQL JSONB column."""
    return Jsonb(metadata)
