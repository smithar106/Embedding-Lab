"""Run golden questions through every model and compute retrieval metrics.

This is the bridge from "models retrieve different chunks" (Phase 2) to "we can
objectively measure retrieval quality" (Phase 3). For each (question, model)
pair it retrieves Top-K results, compares the retrieved document/chunk IDs
against ground truth, and computes Recall@K, reciprocal rank, and latency.
"""
from __future__ import annotations

from evaluation.metrics import (
    aggregate_metrics,
    first_relevant_rank,
    reciprocal_rank,
    recall_at_k,
)
from models.embedding_models import MODELS, embed_query
from retrieval.retriever import retrieve_with_timing


def _relevant_ids(question: dict, level: str) -> list[str]:
    if level == "chunk" and question.get("relevant_chunk_ids"):
        return question["relevant_chunk_ids"]
    return question["relevant_document_ids"]


def _ranked(results: list[dict], level: str) -> list[tuple[str, str]]:
    """Dedupe results to unique (id, title) pairs, preserving rank order.

    At document level, several chunks may belong to the same document — we care
    about *documents* retrieved, so we collapse them to the first occurrence.
    """
    seen: set[str] = set()
    ranked: list[tuple[str, str]] = []
    for r in results:
        key = r["chunk_id"] if level == "chunk" else r["document_id"]
        if key in seen:
            continue
        seen.add(key)
        ranked.append((key, r["title"]))
    return ranked


def evaluate(
    questions: list[dict],
    *,
    models: list[str] | None = None,
    top_k: int = 5,
    relevance_level: str = "document",
) -> dict:
    """Evaluate ``questions`` against every model and return per-question +
    aggregate results (a plain, JSON-serializable dict)."""
    models = models or list(MODELS)

    # Warm the models once so latency reflects warm retrieval, not loading.
    for m in models:
        embed_query("warmup", m)

    per_question: dict = {}
    for q in questions:
        qid = q["question_id"]
        relevant = _relevant_ids(q, relevance_level)
        rel_set = set(relevant)

        model_details: dict = {}
        for model in models:
            results, timing = retrieve_with_timing(q["question"], model, top_k=top_k)
            ranked = _ranked(results, relevance_level)
            retrieved = [key for key, _ in ranked]
            model_details[model] = {
                "retrieved": retrieved,
                "ranks": [
                    {"id": key, "title": title, "relevant": key in rel_set}
                    for key, title in ranked
                ],
                "first_relevant_rank": first_relevant_rank(retrieved, relevant),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "recall_at_5": recall_at_k(retrieved, relevant, 5),
                "latency": timing,
            }

        per_question[qid] = {
            "question": q["question"],
            "relevant": relevant,
            "models": model_details,
        }

    aggregate: dict = {}
    for model in models:
        rows = [
            {
                "retrieved": per_question[q["question_id"]]["models"][model]["retrieved"],
                "relevant": per_question[q["question_id"]]["relevant"],
            }
            for q in questions
        ]
        agg = aggregate_metrics(rows)
        lat = [per_question[q["question_id"]]["models"][model]["latency"] for q in questions]
        n = len(lat)
        agg["avg_embed_ms"] = round(sum(x["embed_query_ms"] for x in lat) / n, 2)
        agg["avg_search_ms"] = round(sum(x["vector_search_ms"] for x in lat) / n, 2)
        agg["avg_total_ms"] = round(sum(x["total_ms"] for x in lat) / n, 2)
        aggregate[model] = agg

    return {"per_question": per_question, "aggregate": aggregate}


def failure_analysis(per_question: dict, models: list[str], top_k: int = 5) -> list[dict]:
    """Highlight questions where models meaningfully differ or miss evidence.

    Returns a list of notable cases:
      - relevant evidence not retrieved in the top-k,
      - relevant evidence ranked below 1,
      - another model retrieved the evidence substantially earlier (>= 2 ranks).
    """
    cases: list[dict] = []
    for qid, qdata in per_question.items():
        ranks = {
            m: qdata["models"][m]["first_relevant_rank"]
            for m in models
        }
        missed = [m for m in models if ranks[m] is None]
        below_one = {m: ranks[m] for m in models if ranks[m] is not None and ranks[m] > 1}

        # Model(s) that retrieved substantially earlier than another.
        ranked_models = [m for m in models if ranks[m] is not None]
        gaps = []
        if len(ranked_models) >= 2:
            best = min(ranked_models, key=lambda m: ranks[m])
            worst = max(ranked_models, key=lambda m: ranks[m])
            if ranks[best] is not None and ranks[worst] is not None and ranks[worst] - ranks[best] >= 2:
                gaps = [{"earlier": best, "later": worst}]

        if missed or below_one or gaps:
            cases.append({
                "question_id": qid,
                "question": qdata["question"],
                "relevant": qdata["relevant"],
                "first_relevant_rank": ranks,
                "missed_in_top_k": missed,
                "below_rank_1": below_one,
                "big_gaps": gaps,
            })
    return cases
