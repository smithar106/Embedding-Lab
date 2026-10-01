"""Tests for the controlled RAG pipeline (mocks retrieval + generation)."""
import json

import generation.rag as ragmod

FAKE_RANKINGS = {
    "minilm": ["a#0", "b#0", "c#0", "d#0", "e#0"],
    "bge": ["b#0", "a#0", "c#0", "d#0", "e#0"],
    "e5": ["a#0", "c#0", "b#0", "d#0", "e#0"],
}


def _fake_retrieve(query, model_name, top_k=5):
    ids = FAKE_RANKINGS[model_name][:top_k]
    results = [
        {
            "chunk_id": cid, "chunk_text": f"text {cid}", "title": cid.split("#")[0],
            "document_id": cid.split("#")[0], "chunk_index": 0, "source": "s",
            "similarity_score": 0.5, "rank": i + 1,
        }
        for i, cid in enumerate(ids)
    ]
    return results, {"embed_query_ms": 1.0, "vector_search_ms": 1.0, "total_ms": 2.0}


def _fake_generate(question, chunks, **kwargs):
    return {
        "answer": f"answer about {chunks[0]['chunk_id']} [{chunks[0]['chunk_id']}]",
        "citations": [chunks[0]["chunk_id"]],
        "model": "deepseek-chat", "temperature": 0.0,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "cost": None, "error": None, "generation_seconds": 0.1,
    }


def test_run_rag_is_serializable(monkeypatch):
    monkeypatch.setattr(ragmod, "retrieve_with_timing", _fake_retrieve)
    monkeypatch.setattr(ragmod, "generate_answer", _fake_generate)
    r = ragmod.run_rag("Q?", "minilm", top_k=5)
    json.dumps(r)  # must not raise
    assert r["embedding_model"] == "minilm"
    assert r["top_k"] == 5
    assert r["retrieved_chunk_ids"] == ["a#0", "b#0", "c#0", "d#0", "e#0"]


def test_same_top_k_across_models(monkeypatch):
    seen = {}

    def record_retrieve(query, model_name, top_k=5):
        seen[model_name] = top_k
        return _fake_retrieve(query, model_name, top_k)

    monkeypatch.setattr(ragmod, "retrieve_with_timing", record_retrieve)
    monkeypatch.setattr(ragmod, "generate_answer", _fake_generate)
    for m in ["minilm", "bge", "e5"]:
        ragmod.run_rag("Q?", m, top_k=5)
    assert seen == {"minilm": 5, "bge": 5, "e5": 5}


def test_generation_model_never_varies_by_embedding_model(monkeypatch):
    # run_rag must NOT pass a per-model generation model — generation is governed
    # by the single env-configured model.
    calls = []

    def record_generate(question, chunks, **kwargs):
        calls.append(kwargs)
        return _fake_generate(question, chunks, **kwargs)

    monkeypatch.setattr(ragmod, "retrieve_with_timing", _fake_retrieve)
    monkeypatch.setattr(ragmod, "generate_answer", record_generate)
    for m in ["minilm", "bge", "e5"]:
        ragmod.run_rag("Q?", m, top_k=5)
    assert all("model" not in kw for kw in calls), "run_rag passed a per-model generation model"
