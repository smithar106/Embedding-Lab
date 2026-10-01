"""
Compare the full RAG pipeline (retrieval + generation) across the three
embedding models for ONE question — so the causal chain is visible:

    embedding model -> ranking -> evidence -> answer

Usage (from the project root, after ingesting the corpus):

    python scripts/compare_rag.py
    python scripts/compare_rag.py --question "Which component stores energy in an electric field instead of chemically?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from models.embedding_models import MODELS, embed_query  # noqa: E402
from generation.rag import run_rag  # noqa: E402

_DISPLAY = {"minilm": "MiniLM", "bge": "BGE", "e5": "E5"}
DEFAULT_QUESTION = "Which component stores energy in an electric field instead of chemically?"


def _usage_str(u) -> str:
    if not u:
        return "n/a"
    return f"{u.get('total_tokens')} tokens (in {u.get('input_tokens')} / out {u.get('output_tokens')})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG across embedding models")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"QUESTION: {args.question}\n")

    # Warm the embedding models once so retrieval latency excludes model loading.
    for m in MODELS:
        embed_query("warmup", m)

    for model in MODELS:
        r = run_rag(args.question, model, top_k=args.top_k)
        print("=" * 72)
        print(f"{_DISPLAY.get(model, model).upper()}")
        print("=" * 72)

        print("Retrieved evidence:")
        for i, ev in enumerate(r["retrieved"], start=1):
            print(f"  {i}. [{ev['chunk_id']}] {ev['title']}")
            print(f"     {ev['chunk_text'][:90]}{'…' if len(ev['chunk_text']) > 90 else ''}")

        print("\nGenerated answer:")
        print(f"  {r['answer']}")

        print("\nCitations:")
        print(f"  {r['citations'] or '(none)'}")

        print("\nLatency:")
        print(f"  Retrieval : {r['retrieval_timing']['total_ms']} ms")
        print(f"  Generation: {r['generation_timing_ms']} ms")
        print(f"  Total     : {r['total_timing_ms']} ms")
        print(f"  Tokens    : {_usage_str(r['usage'])}")
        print()


if __name__ == "__main__":
    main()
