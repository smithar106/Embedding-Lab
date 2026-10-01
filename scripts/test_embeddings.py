"""
Run a single sentence through all three embedding models.

For each model it prints:
    - the short model name
    - the Hugging Face model id
    - the embedding dimension
    - the first 10 values of the resulting vector
    - the embedding latency (first call, which includes the model load/download)

Run from the project root:

    python scripts/test_embeddings.py

The first run downloads each model (a few hundred MB) into the Hugging Face
cache; later runs reuse the cache and are fast.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the `models` package importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.embedding_models import MODELS, embed_query  # noqa: E402

# The sentence passed through every model (treated as a query for demonstration).
SENTENCE = "NASA scientists discovered evidence that liquid water once existed on Mars."


def main() -> None:
    for name, info in MODELS.items():
        print("=" * 64)
        print(f"MODEL                 : {name}")
        print(f"HUGGING FACE MODEL ID : {info.hf_id}")
        print(f"EMBEDDING DIMENSION   : {info.dim}")

        # First encode triggers the model load (and download on the very first
        # run), so this latency includes that one-time cost.
        t0 = time.perf_counter()
        vector = embed_query(SENTENCE, name)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        print("FIRST 10 VECTOR VALUES: [" + "  ".join(f"{x:.6f}" for x in vector[:10]) + "]")
        print(f"EMBEDDING LATENCY     : {cold_ms:.1f} ms  (includes model load)")

        # Warm call — the model is already loaded, so this is pure inference.
        t1 = time.perf_counter()
        embed_query(SENTENCE, name)
        warm_ms = (time.perf_counter() - t1) * 1000.0
        print(f"WARM LATENCY (cached) : {warm_ms:.1f} ms")
    print()


if __name__ == "__main__":
    main()
