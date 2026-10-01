"""Generation layer (Phase 4): shared LLM + controlled RAG pipeline."""
from generation.generator import generate_answer, get_generation_model
from generation.prompts import SYSTEM_PROMPT, build_prompt, parse_citations
from generation.rag import run_rag

__all__ = [
    "generate_answer",
    "get_generation_model",
    "SYSTEM_PROMPT",
    "build_prompt",
    "parse_citations",
    "run_rag",
]
