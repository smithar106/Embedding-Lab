"""Production API tests (FastAPI TestClient; agent/retrieval are mocked)."""
import pytest
from fastapi.testclient import TestClient

import api.app as appmod


def _fake_trace(tool_used=True, error=None, answer="A capacitor [capacitor#0]."):
    return {
        "final_answer": answer,
        "citations": ["capacitor#0"] if tool_used else [],
        "tool_requested": tool_used,
        "tool_results": [
            {"chunk_id": "capacitor#0", "document_id": "capacitor",
             "title": "Capacitors", "text": "...", "rank": 1}
        ],
        "latency": {
            "agent_decision_ms": 10.0, "tool_execution_ms": 20.0,
            "final_generation_ms": 30.0, "total_ms": 60.0,
        },
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        "error": error,
    }


@pytest.fixture
def client(monkeypatch):
    # Avoid loading BGE / hitting the DB during the lifespan and routes.
    monkeypatch.setattr(appmod, "embed_query", lambda *a, **k: [0.0])
    monkeypatch.setattr(appmod, "check_connection", lambda: True)
    monkeypatch.setattr(appmod, "is_model_loaded", lambda name: True)
    monkeypatch.setattr(appmod, "close_pool", lambda: None)
    monkeypatch.setattr(appmod, "run_agent", lambda q: _fake_trace())
    monkeypatch.setattr(
        appmod, "retrieval_search",
        lambda query, top_k=None: {
            "query": query, "embedding_model": "bge", "top_k": top_k or 2,
            "results": [{"chunk_id": "capacitor#0", "document_id": "capacitor",
                         "title": "Capacitors", "text": "...", "rank": 1}],
        },
    )
    with TestClient(appmod.app) as c:
        yield c


def test_health_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"
    assert body["embedding_model"] == "loaded"


def test_health_unhealthy_when_db_down(client, monkeypatch):
    monkeypatch.setattr(appmod, "check_connection", lambda: False)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["database"] == "unavailable"


def test_ask_returns_production_response(client):
    r = client.post("/ask", json={"question": "Which component stores energy in an electric field?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "A capacitor [capacitor#0]."
    assert body["citations"] == ["capacitor#0"]
    assert body["tool_used"] is True
    assert body["retrieved_chunk_ids"] == ["capacitor#0"]
    assert body["latency_ms"]["total"] == 60.0
    assert body["token_usage"]["total_tokens"] == 150
    assert body["request_id"]  # non-empty


def test_ask_invalid_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_generation_error_returns_502(client, monkeypatch):
    monkeypatch.setattr(appmod, "run_agent", lambda q: _fake_trace(error="boom"))
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 502


def test_retrieve_returns_structured_results(client):
    r = client.post("/retrieve", json={"query": "capacitor", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["embedding_model"] == "bge"
    assert body["results"][0]["chunk_id"] == "capacitor#0"


def test_retrieve_invalid_query(client):
    r = client.post("/retrieve", json={"query": ""})
    assert r.status_code == 422


def test_request_id_header_present(client):
    r = client.get("/config")
    assert r.headers.get("x-request-id")


def test_config_returns_model_info(client):
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "generation_model" in body
    assert "embedding_model" in body
    assert "top_k" in body


def test_no_secrets_in_responses(client):
    for path, payload in [("/config", None), ("/health", None),
                          ("/ask", {"question": "hello"})]:
        r = client.post(path, json=payload) if payload else client.get(path)
        text = r.text.lower()
        assert "deepseek_api_key" not in text
        assert "sk-" not in text
        assert "password" not in text
