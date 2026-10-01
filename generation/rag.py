"""A controlled RAG pipeline: retrieval (one embedding model) + shared generation.

    question -> embed_query -> pgvector retrieval -> Top-K chunks
             -> SAME prompt -> SAME LLM -> answer

The ONLY experimental variable is the retrieval path (which embedding model).
Everything downstream — prompt, temperature, generation model, top_k — is held
constant so that any difference in the answer can be traced back to the
difference in retrieved evidence.
"""
from __future__ import annotations

import time

from generation.generator import generate_answer
from retrieval.retriever import retrieve_with_timing


def run_rag(question: str, embedding_model: str, top_k: int = 5) -> dict:
    results, retrieval_timing = retrieve_with_timing(question, embedding_model, top_k)

    chunks = [{"chunk_id": r["chunk_id"], "chunk_text": r["chunk_text"]} for r in results]

    t0 = time.perf_counter()
    gen = generate_answer(question, chunks)
    gen_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "question": question,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "retrieved": results,
        "retrieved_chunk_ids": [r["chunk_id"] for r in results],
        "answer": gen["answer"],
        "citations": gen["citations"],
        "generation_model": gen["model"],
        "temperature": gen["temperature"],
        "usage": gen["usage"],
        "cost": gen["cost"],
        "generation_error": gen["error"],
        "retrieval_timing": retrieval_timing,
        "generation_timing_ms": round(gen_ms, 2),
        "total_timing_ms": round(retrieval_timing["total_ms"] + gen_ms, 2),
    }
