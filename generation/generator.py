"""Shared LLM generation layer.

One model, one prompt, one temperature for every experiment. The generation
model is configured through environment variables and NEVER changes depending
on which embedding model produced the retrieved evidence.

Credentials are read from the environment (see .env.example) — never hard-coded.
"""
from __future__ import annotations

import os
import time

from generation.prompts import build_prompt, parse_citations


def get_generation_model() -> str:
    """The single generation model used for all experiments."""
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def get_generation_temperature() -> float:
    return float(os.environ.get("GENERATION_TEMPERATURE", "0.0"))


def _get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # The `openai` SDK is used purely as a client for DeepSeek's OpenAI-compatible
    # endpoint — no request ever reaches OpenAI.
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _estimate_cost(usage: dict | None) -> float | None:
    """Estimated cost, only if pricing is explicitly configured via env vars.

    We do NOT hard-code provider pricing. If DEEPSEEK_INPUT_COST_PER_M /
    DEEPSEEK_OUTPUT_COST_PER_M are set (USD per million tokens), estimate cost.
    """
    if not usage:
        return None
    in_per_m = os.environ.get("DEEPSEEK_INPUT_COST_PER_M")
    out_per_m = os.environ.get("DEEPSEEK_OUTPUT_COST_PER_M")
    if not in_per_m or not out_per_m:
        return None
    in_cost = float(in_per_m) * usage["input_tokens"] / 1_000_000
    out_cost = float(out_per_m) * usage["output_tokens"] / 1_000_000
    return round(in_cost + out_cost, 8)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    *,
    temperature: float | None = None,
    model: str | None = None,
) -> dict:
    """Generate a grounded answer from the supplied evidence.

    Returns a dict with the answer text, the cited chunk ids (parsed from the
    answer), the generation model, temperature, token usage, and any error.
    """
    model = model or get_generation_model()
    temperature = get_generation_temperature() if temperature is None else temperature
    prompt = build_prompt(question, retrieved_chunks)

    started = time.perf_counter()
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        usage = response.usage
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return {
            "answer": answer,
            "citations": parse_citations(answer),
            "model": model,
            "temperature": temperature,
            "usage": usage_dict,
            "cost": _estimate_cost(usage_dict),
            "error": None,
            "generation_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 — surface failure, never crash the run
        return {
            "answer": None,
            "citations": [],
            "model": model,
            "temperature": temperature,
            "usage": None,
            "cost": None,
            "error": str(exc),
            "generation_seconds": round(time.perf_counter() - started, 3),
        }
