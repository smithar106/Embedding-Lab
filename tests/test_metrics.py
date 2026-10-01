"""Unit tests for the retrieval metrics (pure functions, no DB/models)."""
import pytest

from evaluation.metrics import (
    aggregate_metrics,
    first_relevant_rank,
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_single_relevant_rank1():
    assert recall_at_k(["a", "b", "c"], ["a"], 1) == 1.0


def test_recall_single_relevant_below_rank1():
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0


def test_recall_none_relevant_retrieved():
    assert recall_at_k(["a", "b"], ["z"], 5) == 0.0


def test_recall_multiple_relevant():
    # 2 of 3 relevant items in the top-5.
    assert recall_at_k(["x", "r1", "r2", "y", "z"], ["r1", "r2", "r3"], 5) == pytest.approx(2 / 3)


def test_recall_respects_k_boundary():
    assert recall_at_k(["a", "b", "c", "r2"], ["r2"], 3) == 0.0
    assert recall_at_k(["a", "b", "c", "r2"], ["r2"], 4) == 1.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank_rank1():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_reciprocal_rank_rank2():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_reciprocal_rank_none():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], ["b"]) == 2
    assert first_relevant_rank(["a", "b"], ["z"]) is None


def test_hit_rate():
    assert hit_rate_at_k(["a", "b"], ["a"], 2) == 1.0
    assert hit_rate_at_k(["a", "b"], ["z"], 2) == 0.0


def test_mrr_is_mean_of_reciprocal_ranks():
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a"]},  # rr 1.0
        {"retrieved": ["x", "y"], "relevant": ["y"]},  # rr 0.5
        {"retrieved": ["m", "n"], "relevant": ["z"]},  # rr 0.0
    ]
    agg = aggregate_metrics(rows)
    assert agg["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_aggregate_recall_is_mean():
    rows = [
        {"retrieved": ["a", "b", "c"], "relevant": ["a"]},
        {"retrieved": ["x", "y", "z"], "relevant": ["z"]},
    ]
    agg = aggregate_metrics(rows)
    assert agg["recall_at_1"] == pytest.approx(0.5)
    assert agg["recall_at_5"] == pytest.approx(1.0)


def test_aggregate_empty():
    agg = aggregate_metrics([])
    assert agg["mrr"] == 0.0 and agg["recall_at_1"] == 0.0
