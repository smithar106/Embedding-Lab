"""
Run the full indexing pipeline:

    load documents -> chunk ONCE -> embed with MiniLM/BGE/E5 -> store in pgvector

Usage (from the project root):

    python scripts/ingest_dataset.py                 # index data/raw
    python scripts/ingest_dataset.py --path data/raw --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project packages importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.indexer import index_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a dataset into pgvector")
    parser.add_argument("--path", default="data/raw", help="file or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--reset", action="store_true", help="wipe the index first")
    args = parser.parse_args()

    summary = index_dataset(
        args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
