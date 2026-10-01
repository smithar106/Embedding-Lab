"""Tests for the evaluator's orchestration + result serialization.

These use a fake ``retrieve_with_timing`` so they exercise the metric
aggregation and per-question bookkeeping deterministically, without a database
or loaded models.
"""
import json

import evaluation.evaluator as ev
from evaluation.evaluator import evaluate, failure_analysis

# Deterministic per-model rankings (chunk_id -> document_id = first segment).
FAKE_RANKINGS = {
    "minilm": ["a#0", "b#0", "c#0", "d#0", "e#0"],
    "bge": ["b#0", "a#0", "c#0", "d#0", "e#0"],
    "e5": ["a#0", "b#0", "c#0", "d#0", "e#0"],
}


def _fake_retrieve_with_timing(query, model_name, top_k=5):
    ids = FAKE_RANKINGS[model_name][:top_k]
    results = [
        {"chunk_id": cid, "document_id": cid.split("#")[0], "title": cid.split("#")[0]}
        for cid in ids
    ]
    timing = {"embed_query_ms": 1.0, "vector_search_ms": 1.0, "total_ms": 2.0}
    return results, timing


def test_all_models_evaluated_on_identical_questions(monkeypatch):
    monkeypatch.setattr(ev, "retrieve_with_timing", _fake_retrieve_with_timing)
    questions = [
        {"question_id": "q1", "question": "?", "relevant_document_ids": ["a"]},
        {"question_id": "q2", "question": "?", "relevant_document_ids": ["b"]},
    ]
    result = evaluate(questions, top_k=5)
    assert set(result["aggregate"]) == {"minilm", "bge", "e5"}
    assert set(result["per_question"]) == {"q1", "q2"}
    for q in questions:
        assert set(result["per_question"][q["question_id"]]["models"]) == {"minilm", "bge", "e5"}


def test_per_question_first_relevant_rank(monkeypatch):
    monkeypatch.setattr(ev, "retrieve_with_timing", _fake_retrieve_with_timing)
    questions = [{"question_id": "q1", "question": "?", "relevant_document_ids": ["b"]}]
    result = evaluate(questions, top_k=5)
    # minilm has b at rank 2; bge has b at rank 1.
    assert result["per_question"]["q1"]["models"]["minilm"]["first_relevant_rank"] == 2
    assert result["per_question"]["q1"]["models"]["bge"]["first_relevant_rank"] == 1
    assert result["per_question"]["q1"]["models"]["minilm"]["reciprocal_rank"] == 0.5


def test_aggregate_recall_mrr(monkeypatch):
    monkeypatch.setattr(ev, "retrieve_with_timing", _fake_retrieve_with_timing)
    questions = [{"question_id": "q1", "question": "?", "relevant_document_ids": ["b"]}]
    result = evaluate(questions, top_k=5)
    # minilm: b at rank 2 -> rr 0.5, recall@5 = 1.0, recall@1 = 0.0
    a = result["aggregate"]["minilm"]
    assert a["recall_at_1"] == 0.0
    assert a["recall_at_5"] == 1.0
    assert a["mrr"] == 0.5
    # bge: b at rank 1
    assert result["aggregate"]["bge"]["mrr"] == 1.0


def test_result_is_serializable_and_deterministic(monkeypatch):
    monkeypatch.setattr(ev, "retrieve_with_timing", _fake_retrieve_with_timing)
    questions = [{"question_id": "q1", "question": "?", "relevant_document_ids": ["a", "z"]}]
    r1 = evaluate(questions, top_k=5)
    r2 = evaluate(questions, top_k=5)
    assert r1["aggregate"] == r2["aggregate"]
    json.dumps(r1)  # must not raise
    # recall@5 = 1 relevant found out of 2 -> 0.5
    assert r1["per_question"]["q1"]["models"]["minilm"]["recall_at_5"] == 0.5


def test_failure_analysis_detects_miss(monkeypatch):
    monkeypatch.setattr(ev, "retrieve_with_timing", _fake_retrieve_with_timing)
    questions = [{"question_id": "q1", "question": "?", "relevant_document_ids": ["z"]}]
    result = evaluate(questions, top_k=5)
    cases = failure_analysis(result["per_question"], ["minilm", "bge", "e5"], top_k=5)
    assert any(c["question_id"] == "q1" for c in cases)
    case = next(c for c in cases if c["question_id"] == "q1")
    assert case["missed_in_top_k"] == ["minilm", "bge", "e5"]
