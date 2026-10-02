"""Tests for the retrieval tool boundary."""
import pytest

import tools.retrieval_tool as rt
from tools.retrieval_tool import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    retrieval_search,
)


def test_defaults_are_experimentally_selected():
    assert DEFAULT_EMBEDDING_MODEL == "bge"
    assert DEFAULT_TOP_K == 2


def test_retrieval_search_returns_structured_output(monkeypatch):
    fake = [
        {"chunk_id": "a#0", "document_id": "a", "title": "A", "chunk_text": "text A",
         "similarity_score": 0.9, "rank": 1},
        {"chunk_id": "b#0", "document_id": "b", "title": "B", "chunk_text": "text B",
         "similarity_score": 0.8, "rank": 2},
    ]
    monkeypatch.setattr(rt, "retrieve", lambda query, model, top_k: fake[:top_k])
    out = retrieval_search("q", top_k=2)

    assert out["query"] == "q"
    assert out["embedding_model"] == "bge"
    assert out["top_k"] == 2
    assert out["results"][0]["chunk_id"] == "a#0"
    assert out["results"][0]["rank"] == 1
    assert out["results"][0]["text"] == "text A"
    # raw similarity scores are kept internal (not exposed to the agent)
    assert "similarity_score" not in out["results"][0]


def test_retrieval_search_validation():
    with pytest.raises(ValueError):
        retrieval_search("")
    with pytest.raises(ValueError):
        retrieval_search("q", top_k=0)
    with pytest.raises(ValueError):
        retrieval_search("q", top_k="two")


def test_tool_schema_exposes_only_query_and_top_k():
    props = rt.TOOL_SCHEMA["function"]["parameters"]["properties"]
    assert set(props) == {"query", "top_k"}
    assert "embedding_model" not in props  # kept internal to the tool
