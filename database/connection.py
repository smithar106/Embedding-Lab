"""PostgreSQL + pgvector connection helpers.

We use psycopg (psycopg 3) directly — no ORM — so the SQL and the pgvector
operators stay visible, which suits an educational project.
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

# Local Homebrew Postgres default (no password, current OS user). Override by
# setting DATABASE_URL (see .env.example).
DEFAULT_DATABASE_URL = "postgresql://localhost:5432/embedding_lab"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> psycopg.Connection:
    """Open a connection and register the pgvector type adapter.

    ``register_vector(conn)`` teaches psycopg how to convert a Python list /
    numpy array into pgvector's ``vector`` type, so we can pass embeddings
    straight into SQL parameters.
    """
    url = os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def init_db(conn: psycopg.Connection) -> None:
    """Create the schema (idempotent)."""
    conn.execute(_SCHEMA_PATH.read_text())
    conn.commit()


def reset_db(conn: psycopg.Connection) -> None:
    """Drop all rows (used by ``ingest --reset``)."""
    conn.execute("TRUNCATE documents, chunks CASCADE")
    conn.commit()
