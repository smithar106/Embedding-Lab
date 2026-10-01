"""Retrieve the Top-K most similar chunks for a query, using ONE model.

QUERY TIME is deliberately cheap: we embed the query once (with the model's
query prefix), then run a pgvector cosine-similarity search against the
embeddings we stored at indexing time. No documents are re-embedded here.

CRITICAL RULE — embedding spaces cannot be mixed. A MiniLM query embedding is a
point in MiniLM's 384-dimensional space; BGE and E5 live in different 768-dim
spaces (and even BGE vs E5 are different coordinate systems). So a query must
be searched ONLY against the same model's stored embeddings. ``MODEL_TABLES``
enforces this by routing each model to its own table.
"""
from __future__ import annotations

import time

from database import MODEL_TABLES
from database.connection import get_connection
from models.embedding_models import embed_query


def retrieve(query: str, model_name: str, top_k: int = 5) -> list[dict]:
    """Return the Top-K chunks for ``query`` using ``model_name``.

    Each result is a dict: rank, chunk_id, document_id, chunk_index, title,
    chunk_text, similarity_score, source.

    ``similarity_score`` is cosine similarity (1 - pgvector cosine distance).
    """
    if model_name not in MODEL_TABLES:
        raise KeyError(f"unknown model '{model_name}'. Choose from {list(MODEL_TABLES)}")

    table = MODEL_TABLES[model_name]

    # 1. Embed the query with the model's OWN query-side handling (prefixes).
    query_vector = embed_query(query, model_name)

    conn = get_connection()
    try:
        # 2. Cosine-similarity search: ``<=>`` is pgvector's cosine *distance*,
        #    so ``1 - distance`` = similarity. Order ascending distance = desc sim.
        sql = f"""
            SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                   c.title, c.source,
                   1 - (e.embedding <=> %s) AS similarity
            FROM {table} e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "rank": rank,
            "chunk_id": row[0],
            "document_id": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3],
            "title": row[4],
            "source": row[5],
            "similarity_score": round(float(row[6]), 4),
        })
    return results


def retrieve_with_timing(query: str, model_name: str, top_k: int = 5) -> tuple[list[dict], dict]:
    """Like ``retrieve`` but also reports embed/search/total latency (warm)."""
    table = MODEL_TABLES[model_name]
    conn = get_connection()
    try:
        # Embed timing (model already warm — do one throwaway call first).
        t0 = time.perf_counter()
        query_vector = embed_query(query, model_name)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        sql = f"""
            SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                   c.title, c.source,
                   1 - (e.embedding <=> %s) AS similarity
            FROM {table} e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """
        t1 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, top_k))
            rows = cur.fetchall()
        search_ms = (time.perf_counter() - t1) * 1000.0
    finally:
        conn.close()

    total_ms = embed_ms + search_ms
    results = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "rank": rank,
            "chunk_id": row[0],
            "document_id": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3],
            "title": row[4],
            "source": row[5],
            "similarity_score": round(float(row[6]), 4),
        })
    return results, {
        "embed_query_ms": round(embed_ms, 2),
        "vector_search_ms": round(search_ms, 2),
        "total_ms": round(total_ms, 2),
    }
