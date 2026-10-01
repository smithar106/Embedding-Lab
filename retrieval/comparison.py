"""Compare retrieval across the three embedding models.

This is the "results" half of Embedding-Lab: given ONE query, run it through
MiniLM / BGE / E5 independently, then compare rankings, overlap, and latency.

IMPORTANT — do not compare raw similarity SCORES across models. The scores are
not directly comparable: each model lives in a different embedding space with a
different similarity distribution (e.g. E5 typically returns higher raw cosine
values than MiniLM for the same match). We compare *rankings*, *retrieved
chunks*, *overlap*, and *latency* — not the raw numbers.
"""
from __future__ import annotations

from database import MODEL_TABLES
from models.embedding_models import MODELS


def retrieve_all(query: str, top_k: int = 5) -> dict:
    """Run the query through every model and return {model: (results, timing)}.

    Models are warmed first (one throwaway embedding each) so the timings below
    reflect *warm* retrieval, not the one-time model load/download.
    """
    from models.embedding_models import embed_query
    from retrieval.retriever import retrieve_with_timing

    # Warm each model so loading/download is not counted in retrieval latency.
    for name in MODEL_TABLES:
        embed_query("warmup", name)

    out = {}
    for name in MODEL_TABLES:
        results, timing = retrieve_with_timing(query, name, top_k=top_k)
        out[name] = {"results": results, "timing": timing}
    return out


def _chunk_ids(model_out: dict) -> set[str]:
    return {r["chunk_id"] for r in model_out["results"]}


def compute_overlap(all_results: dict, top_k: int = 5) -> dict:
    """Pairwise overlap (% of top_k shared) and the chunks shared by all models."""
    models = list(all_results.keys())
    sets = {m: _chunk_ids(all_results[m]) for m in models}

    pairwise = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            shared = sets[a] & sets[b]
            pairwise[f"{a}_vs_{b}"] = {
                "shared": len(shared),
                "overlap_pct": round(len(shared) / top_k * 100, 1),
                "chunk_ids": sorted(shared),
            }

    shared_all = set.intersection(*[sets[m] for m in models]) if models else set()
    return {"pairwise": pairwise, "shared_all": sorted(shared_all)}
