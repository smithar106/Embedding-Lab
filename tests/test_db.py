"""Integration tests against PostgreSQL + pgvector (skipped if unavailable).

NOTE: these tests run ``reset=True`` and re-index a tiny document, so they wipe
and rewrite the index. Run the demo dataset again afterwards with:

    python scripts/ingest_dataset.py
"""
import tempfile
from pathlib import Path

import pytest

from database import MODEL_TABLES
from database.connection import get_connection


@pytest.fixture(scope="module")
def db_available():
    try:
        from database.connection import get_connection, init_db

        conn = get_connection()
        init_db(conn)  # ensure the schema exists for the tests below
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        pytest.skip("PostgreSQL/pgvector not available")


def _write_doc(tmp_path: Path) -> Path:
    f = tmp_path / "testdoc.txt"
    f.write_text("Mars water evidence\n" + ("Water on Mars is a key question. " * 60))
    return f


def test_vector_dimensions_in_schema_match_models(db_available):
    conn = get_connection()
    try:
        for model_name, table in MODEL_TABLES.items():
            row = conn.execute(
                """
                SELECT a.atttypmod AS dim
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = %s AND a.attname = 'embedding'
                """,
                (table,),
            ).fetchone()
            expected = {"minilm": 384, "bge": 768, "e5": 768}[model_name]
            assert row[0] == expected, f"{table} is {row[0]}-dim, expected {expected}"
    finally:
        conn.close()


def test_indexing_and_retrieval(tmp_path, db_available):
    from ingestion.indexer import index_dataset

    summary = index_dataset(str(_write_doc(tmp_path)), reset=True, verbose=False)
    assert summary["chunks_created"] >= 1
    assert summary["embeddings_generated"] == summary["chunks_created"] * 3  # 3 models

    conn = get_connection()
    try:
        # Every model indexes the exact same chunk ids.
        ids_per_model = {}
        for model, table in MODEL_TABLES.items():
            rows = conn.execute(f"SELECT chunk_id FROM {table} ORDER BY chunk_id").fetchall()
            ids_per_model[model] = {r[0] for r in rows}
        assert ids_per_model["minilm"] == ids_per_model["bge"] == ids_per_model["e5"]
        assert len(ids_per_model["minilm"]) == summary["chunks_created"]
    finally:
        conn.close()


def test_retrieval_top_k_count(tmp_path, db_available):
    from retrieval.retriever import retrieve

    results = retrieve("water on Mars", "minilm", top_k=3)
    assert 0 < len(results) <= 3
    for r in results:
        assert set(r.keys()) >= {"rank", "chunk_id", "document_id", "title", "chunk_text", "similarity_score", "source"}


def test_reingestion_is_idempotent(tmp_path, db_available):
    from ingestion.indexer import index_dataset

    path = str(_write_doc(tmp_path))
    index_dataset(path, reset=True, verbose=False)
    summary2 = index_dataset(path, verbose=False)

    assert summary2["chunks_changed"] == 0
    assert summary2["embeddings_generated"] == 0  # nothing re-embedded
    assert summary2["chunks_unchanged"] == summary2["chunks_created"]
