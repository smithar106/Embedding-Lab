"""Tests for the shared generation layer (mocks the OpenAI client — no API)."""
import generation.generator as gen
from generation.prompts import SYSTEM_PROMPT, build_prompt


class _FakeUsage:
    prompt_tokens = 12
    completion_tokens = 8
    total_tokens = 20


class _FakeMessage:
    content = "Capacitors store energy in an electric field [capacitor#0]."


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_get_generation_model_default(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert gen.get_generation_model() == "deepseek-chat"


def test_get_generation_model_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    assert gen.get_generation_model() == "deepseek-chat"


def test_generate_answer_returns_answer_citations_usage(monkeypatch):
    monkeypatch.setattr(gen, "_get_client", lambda: _FakeClient())
    out = gen.generate_answer("Q?", [{"chunk_id": "capacitor#0", "chunk_text": "..."}])
    assert out["error"] is None
    assert out["citations"] == ["capacitor#0"]
    assert out["usage"]["total_tokens"] == 20
    assert out["usage"]["input_tokens"] == 12
    assert out["usage"]["output_tokens"] == 8


def test_generate_answer_failure_is_handled(monkeypatch):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(gen, "_get_client", boom)
    out = gen.generate_answer("Q?", [])
    assert out["error"] == "boom"
    assert out["answer"] is None
    assert out["citations"] == []


def test_prompt_is_deterministic_and_constant_system(monkeypatch):
    chunks = [{"chunk_id": "a#0", "chunk_text": "text"}]
    p1 = build_prompt("Q", chunks)
    p2 = build_prompt("Q", chunks)
    assert p1 == p2
    assert p1[0]["content"] == SYSTEM_PROMPT
    assert "a#0" in p1[1]["content"]
    assert "Q" in p1[1]["content"]
