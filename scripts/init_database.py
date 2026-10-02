"""
Deliberate database initialization command.

    1. connect using DATABASE_URL
    2. enable pgvector
    3. create the schema
    4. verify expected tables/indexes
    5. exit cleanly

This is a ONE-TIME (or per-migration) operation — the API does NOT run it on
every startup, and it never drops/recreates tables.

Usage (from the project root):

    python -m scripts.init_database
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from database.connection import (  # noqa: E402
    close_pool,
    get_connection,
    init_db,
    verify_schema,
)


def main() -> None:
    print("Initializing database…")
    with get_connection() as conn:
        init_db(conn)  # idempotent: CREATE EXTENSION IF NOT EXISTS vector + schema
    print("Schema created (idempotent).")

    status = verify_schema()
    close_pool()
    if status["ok"]:
        print("Verification OK: all expected tables present.")
    else:
        print(f"Verification FAILED: missing {status['missing']} error={status['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
