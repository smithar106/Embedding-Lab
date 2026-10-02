"""Library metadata — aggregated stats about the corpus, for the frontend.

The frontend renders "what does this library cover?" from THIS data, so it never
hardcodes collection names, topic tags, source organizations, or counts. When
the corpus grows (new collections, new sources), the endpoint and the page
update together with no frontend change.
"""
from __future__ import annotations

from database.connection import get_connection
from models.embedding_models import MODELS

# Collection slug -> display name. Kept here (not in the frontend) so the UI
# always shows the canonical label for whatever is in the database.
COLLECTION_LABELS = {
    "climate": "Climate",
    "energy": "Energy",
    "food-agriculture": "Food & Agriculture",
    "water": "Water",
    "cities": "Cities",
    "global-development": "Global Development",
    "uncategorized": "Other",
}

# Model key -> display name for the cataloging-system section.
MODEL_LABELS = {"minilm": "MiniLM", "bge": "BGE", "e5": "E5"}


def get_library_stats() -> dict:
    """Return collection/source/model summaries straight from the database."""
    with get_connection() as conn:
        total_sources = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        total_passages = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

        collections = conn.execute(
            """
            SELECT d.collection,
                   count(DISTINCT d.document_id) AS documents,
                   count(c.chunk_id) AS passages
            FROM documents d
            JOIN chunks c ON c.document_id = d.document_id
            GROUP BY d.collection
            ORDER BY d.collection
            """
        ).fetchall()

        # Distinct topics per collection (topics is TEXT[]; unnest to aggregate).
        topics = conn.execute(
            """
            SELECT collection, array_agg(DISTINCT topic ORDER BY topic) AS topics
            FROM documents, unnest(topics) AS topic
            GROUP BY collection
            """
        ).fetchall()
        topics_by_collection = {row[0]: list(row[1]) for row in topics}

        organizations = conn.execute(
            """
            SELECT d.source_organization,
                   count(DISTINCT d.document_id) AS documents,
                   count(c.chunk_id) AS passages
            FROM documents d
            JOIN chunks c ON c.document_id = d.document_id
            GROUP BY d.source_organization
            ORDER BY documents DESC, d.source_organization
            """
        ).fetchall()

    return {
        "totals": {
            "sources": total_sources,
            "passages": total_passages,
            "collections": len(collections),
        },
        "collections": [
            {
                "id": row[0],
                "name": COLLECTION_LABELS.get(row[0], row[0]),
                "documents": row[1],
                "passages": row[2],
                "topics": topics_by_collection.get(row[0], []),
            }
            for row in collections
        ],
        "organizations": [
            {"name": row[0], "documents": row[1], "passages": row[2]}
            for row in organizations
        ],
        "models": [
            {
                "key": info.key,
                "name": MODEL_LABELS.get(info.key, info.key),
                "hf_id": info.hf_id,
                "dim": info.dim,
                "use": info.use,
            }
            for info in MODELS.values()
        ],
    }
