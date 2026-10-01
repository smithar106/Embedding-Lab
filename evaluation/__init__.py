"""Retrieval evaluation (Phase 3)."""
from evaluation.dataset import load_golden_questions, validate_question
from evaluation.evaluator import evaluate, failure_analysis
from evaluation.metrics import (
    aggregate_metrics,
    first_relevant_rank,
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "load_golden_questions",
    "validate_question",
    "evaluate",
    "failure_analysis",
    "aggregate_metrics",
    "first_relevant_rank",
    "hit_rate_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
