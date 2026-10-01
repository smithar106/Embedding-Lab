"""Retrieval metrics — implemented from scratch in readable Python.

Every metric is a pure function over two lists of IDs::

    retrieved_ids : the ranked IDs the system returned (in rank order)
    relevant_ids  : the ground-truth IDs marked relevant for the question

They are deliberately small and self-contained so you can read exactly what
each one computes — nothing is hidden behind a large evaluation framework.
"""
from __future__ import annotations


def recall_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    """Fraction of the relevant items that appear within the top-k results.

        recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    For a question with a single relevant document this is 0.0 or 1.0; for
    multiple relevant documents it measures how much of the relevant evidence
    was found.
    """
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    """1.0 if ANY relevant item is in the top-k, else 0.0 (binary per question)."""
    if not relevant_ids:
        return 0.0
    return 1.0 if (set(retrieved_ids[:k]) & set(relevant_ids)) else 0.0


def first_relevant_rank(retrieved_ids, relevant_ids):
    """1-based rank of the first relevant item, or None if none was retrieved."""
    rel = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return i
    return None


def reciprocal_rank(retrieved_ids, relevant_ids) -> float:
    """1 / rank of the first relevant item (0.0 if none found).

    Example: relevant doc at rank 2 -> RR = 1/2 = 0.5.
    """
    rank = first_relevant_rank(retrieved_ids, relevant_ids)
    return 1.0 / rank if rank else 0.0


def aggregate_metrics(rows: list[dict]) -> dict:
    """Average per-question metrics across a list of
    ``{"retrieved": [...], "relevant": [...]}`` rows."""
    n = len(rows)
    if n == 0:
        return {
            "recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0,
            "mrr": 0.0, "hit_rate_at_5": 0.0,
        }
    return {
        "recall_at_1": sum(recall_at_k(r["retrieved"], r["relevant"], 1) for r in rows) / n,
        "recall_at_3": sum(recall_at_k(r["retrieved"], r["relevant"], 3) for r in rows) / n,
        "recall_at_5": sum(recall_at_k(r["retrieved"], r["relevant"], 5) for r in rows) / n,
        "mrr": sum(reciprocal_rank(r["retrieved"], r["relevant"]) for r in rows) / n,
        "hit_rate_at_5": sum(hit_rate_at_k(r["retrieved"], r["relevant"], 5) for r in rows) / n,
    }
