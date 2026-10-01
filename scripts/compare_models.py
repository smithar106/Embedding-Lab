"""
Compare how each model ranks the same passages for a query.

For every model this script:
    1. embeds the query,
    2. embeds each passage,
    3. computes cosine similarity between the query and every passage,
    4. ranks the passages by similarity and prints the ranking.

The ranking shows which passage each model thinks is *most semantically
similar* to the query — i.e. the passage a retrieval system would return first.

Run from the project root:

    python scripts/compare_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the `models` package importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.embedding_models import (  # noqa: E402
    MODELS,
    cosine_similarity,
    embed_document,
    embed_query,
)

QUERY = "What evidence is there that Mars once had water?"

PASSAGES = [
    "Orbital observations identified minerals that typically form in the presence of liquid water.",
    "The James Webb Space Telescope observes distant galaxies using infrared light.",
    "Mars has polar ice caps composed of water ice and carbon dioxide ice.",
    "NASA launched the Voyager spacecraft to study the outer planets.",
]


def rank_passages(model_name: str) -> list[tuple[float, str]]:
    """Embed the query + passages and return (similarity, passage) sorted desc."""
    q = embed_query(QUERY, model_name)

    scored = []
    for passage in PASSAGES:
        v = embed_document(passage, model_name)
        scored.append((cosine_similarity(q, v), passage))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def main() -> None:
    print(f"QUERY: {QUERY!r}\n")

    for name, info in MODELS.items():
        print("=" * 78)
        print(f"MODEL: {name}  ({info.hf_id})")
        print("-" * 78)

        ranked = rank_passages(name)
        for rank, (sim, passage) in enumerate(ranked, start=1):
            winner = "  <-- most similar" if rank == 1 else ""
            print(f"  {rank}.  sim={sim:+.4f}   {passage}{winner}")
        print()


if __name__ == "__main__":
    main()
