"""PostgreSQL + pgvector connection management.

We use psycopg (psycopg 3) directly — no ORM — so the SQL and the pgvector
operators stay visible. Connections come from a small ``psycopg_pool`` pool so
we don't open an uncontrolled new connection per internal operation.

``DATABASE_URL`` fully determines the database. The local Homebrew default is
only a fallback for local development; production uses the Railway-injected URL.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DEFAULT_DATABASE_URL = "postgresql://localhost:5432/embedding_lab"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_REQUIRED_TABLES = ["documents", "chunks", "chunk_embeddings_minilm",
                    "chunk_embeddings_bge", "chunk_embeddings_e5"]

_pool: ConnectionPool | None = None


def _url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        # configure=register_vector teaches every pooled connection how to adapt
        # Python lists / numpy arrays into pgvector's `vector` type.
        _pool = ConnectionPool(
            _url(),
            min_size=1,
            max_size=10,
            configure=register_vector,
            open=False,
            timeout=10,
        )
        # Open lazily in the background; connections are established on demand.
        _pool.open(wait=False)
    return _pool


@contextmanager
def get_connection():
    """Yield a pooled connection (returns to the pool on exit)."""
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_db(conn: psycopg.Connection | None = None) -> None:
    """Create the schema (idempotent).

    When ``conn`` is omitted, this opens a DIRECT connection (not the pool)
    because the schema creates the pgvector extension, and the pool's
    ``configure=register_vector`` callback requires the ``vector`` type to
    already exist — a chicken-and-egg on a fresh database.
    """
    if conn is None:
        conn = psycopg.connect(_url(), connect_timeout=5)
        try:
            conn.execute(_SCHEMA_PATH.read_text())
            conn.commit()
        finally:
            conn.close()
    else:
        conn.execute(_SCHEMA_PATH.read_text())
        conn.commit()


def reset_db(conn: psycopg.Connection) -> None:
    """Drop all rows (used by ``ingest --reset``)."""
    conn.execute("TRUNCATE documents, chunks CASCADE")
    conn.commit()


def check_connection() -> bool:
    """Return True if the database is reachable (used by /health).

    Uses a short-lived direct connection (not the pool) so /health fails fast
    when the database is down.
    """
    try:
        conn = psycopg.connect(_url(), connect_timeout=3)
        try:
            conn.execute("SELECT 1")
            return True
        finally:
            conn.close()
    except Exception:
        return False


def verify_schema() -> dict:
    """Check that the expected tables exist. Returns a status dict."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        existing = {r[0] for r in rows}
        missing = [t for t in _REQUIRED_TABLES if t not in existing]
        return {"ok": not missing, "missing": missing, "error": None}
    except Exception as exc:
        return {"ok": False, "missing": [], "error": str(exc)}
