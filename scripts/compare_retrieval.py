"""
Compare retrieval across all three models for ONE query: rankings, latency,
and chunk overlap.

Usage (from the project root, after running ingest_dataset.py):

    python scripts/compare_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.comparison import compute_overlap, retrieve_all  # noqa: E402

QUERY = "What evidence is there that Mars once had liquid water?"
TOP_K = 5


def main() -> None:
    print(f"QUERY: {QUERY!r}\n")

    all_results = retrieve_all(QUERY, top_k=TOP_K)

    # ---- Per-model sections -------------------------------------------------
    for model_name, out in all_results.items():
        t = out["timing"]
        print("=" * 70)
        print(f"{model_name.upper()}")
        print("=" * 70)
        print(f"  Embedding query : {t['embed_query_ms']} ms")
        print(f"  Vector search   : {t['vector_search_ms']} ms")
        print(f"  Total retrieval : {t['total_ms']} ms\n")
        for r in out["results"]:
            print(f"  {r['rank']}.  {r['chunk_id']}")
            print(f"      Score: {r['similarity_score']:.4f}")
            print(f"      Title: {r['title']}")
            print(f"      Text : {r['chunk_text'][:100]}{'…' if len(r['chunk_text']) > 100 else ''}")
        print()

    # ---- Overlap ------------------------------------------------------------
    overlap = compute_overlap(all_results, top_k=TOP_K)
    print("=" * 70)
    print("OVERLAP")
    print("=" * 70)
    for label, o in overlap["pairwise"].items():
        print(f"  {label:15}: {o['shared']}/{TOP_K} shared chunks  ({o['overlap_pct']}% overlap)")
    print(f"\n  Shared across all three models:")
    for chunk_id in overlap["shared_all"]:
        print(f"    {chunk_id}")
    if not overlap["shared_all"]:
        print("    (none)")

    print("\nNote: raw similarity scores are NOT comparable across models —")
    print("each model lives in its own embedding space with its own score")
    print("distribution. Compare rankings, overlap, and latency instead.")


if __name__ == "__main__":
    main()
