"""Tests for the minimal agent (mocks the DeepSeek client — no API)."""
import json

import agent.agent as agentmod
from agent.agent import run_agent


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ToolCall:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses):
        self.chat = type("C", (), {"completions": _Completions(responses)})()


def _tool_call_client(final_answer):
    return _Client([
        _Resp(_Msg(content=None, tool_calls=[
            _ToolCall("retrieval_search", json.dumps({"query": "q?", "top_k": 2}))
        ])),
        _Resp(_Msg(content=final_answer)),
    ])


def _direct_client(answer):
    return _Client([_Resp(_Msg(content=answer))])


FAKE_RESULT = {
    "query": "q?", "embedding_model": "bge", "top_k": 2,
    "results": [
        {"chunk_id": "capacitor#0", "document_id": "capacitor", "title": "Capacitors",
         "text": "a capacitor stores energy in an electric field", "rank": 1},
    ],
}


def _patch(monkeypatch, client, final):
    monkeypatch.setattr(agentmod, "get_client", lambda: client)
    monkeypatch.setattr(agentmod, "get_generation_model", lambda: "deepseek-chat")
    monkeypatch.setattr(agentmod, "retrieval_search",
                        lambda query, top_k, embedding_model: FAKE_RESULT)


def test_agent_calls_tool_and_validates_citations(monkeypatch):
    _patch(monkeypatch, _tool_call_client("A capacitor [capacitor#0]."), None)
    t = run_agent("Which component stores energy in an electric field?")
    assert t["tool_requested"] is True
    assert t["tool_name"] == "retrieval_search"
    assert t["tool_arguments"] == {"query": "q?", "top_k": 2}
    assert t["tool_results"] == FAKE_RESULT["results"]
    assert t["final_answer"] == "A capacitor [capacitor#0]."
    assert t["citations"] == ["capacitor#0"]
    assert t["citation_validity"]["ratio"] == 1.0


def test_agent_skips_tool_when_not_needed(monkeypatch):
    monkeypatch.setattr(agentmod, "get_client", lambda: _direct_client("Hello!"))
    monkeypatch.setattr(agentmod, "get_generation_model", lambda: "deepseek-chat")
    called = []
    monkeypatch.setattr(agentmod, "retrieval_search", lambda *a, **kw: called.append(1))
    t = run_agent("Say hello.")
    assert t["tool_requested"] is False
    assert t["final_answer"] == "Hello!"
    assert called == []  # the tool was never executed


def test_tool_result_returns_to_model(monkeypatch):
    client = _tool_call_client("A capacitor [capacitor#0].")
    _patch(monkeypatch, client, None)
    run_agent("q?")
    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert any(m["role"] == "tool" for m in second_call_messages)


def test_hallucinated_citation_detected(monkeypatch):
    _patch(monkeypatch, _tool_call_client("Answer [zzz#9]."), None)
    t = run_agent("q?")
    assert t["citation_validity"]["ratio"] == 0.0
    assert t["citation_validity"]["invalid_citations"] == ["zzz#9"]


def test_trace_is_serializable(monkeypatch):
    _patch(monkeypatch, _tool_call_client("A capacitor [capacitor#0]."), None)
    t = run_agent("q?")
    json.dumps(t)  # must not raise
    assert t["latency"]["total_ms"] >= 0
