"""
Run ONE query through retrieval for each model and print the Top-K results.

Usage (from the project root, after running ingest_dataset.py):

    python scripts/test_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import MODEL_TABLES  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402

QUERY = "What evidence is there that Mars once had liquid water?"
TOP_K = 5


def main() -> None:
    print(f"QUERY: {QUERY!r}\n")

    for model_name in MODEL_TABLES:
        print("=" * 70)
        print(f"{model_name.upper()}")
        print("-" * 70)
        results = retrieve(QUERY, model_name, top_k=TOP_K)
        for r in results:
            print(f"  {r['rank']}.  {r['chunk_id']}  (score={r['similarity_score']:.4f})")
            print(f"      title: {r['title']}")
            print(f"      text : {r['chunk_text'][:110]}{'…' if len(r['chunk_text']) > 110 else ''}")
        print()


if __name__ == "__main__":
    main()
