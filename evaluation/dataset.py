"""Load and validate the golden evaluation dataset.

A golden question has the shape::

    {
      "question_id": "q001",
      "question": "What evidence suggests Mars once had liquid water?",
      "relevant_document_ids": ["mars-liquid-water"],
      "relevant_chunk_ids": ["mars-liquid-water#0"],   # optional, chunk-level
      "notes": "..."                                    # optional
    }

Two relevance levels are supported (see the README):

- **Document-level** (`relevant_document_ids`) — "did we retrieve evidence from
  the correct document?" This is the default and is sufficient for most
  questions.
- **Chunk-level** (`relevant_chunk_ids`) — "did we retrieve the exact relevant
  passage?" Optional, for fine-grained evaluation later.

Ground truth is authored by hand against the source documents — the evaluator
never asks a model to decide relevance.
"""
from __future__ import annotations

import json
from pathlib import Path

_REQUIRED_KEYS = {"question_id", "question", "relevant_document_ids"}


def validate_question(q: dict, *, index: int | None = None) -> dict:
    where = f"question #{index}" if index is not None else f"question {q.get('question_id', '?')}"
    if not isinstance(q, dict):
        raise ValueError(f"{where} must be a JSON object")

    missing = _REQUIRED_KEYS - set(q)
    if missing:
        raise ValueError(f"{where} is missing required keys: {sorted(missing)}")

    rel = q["relevant_document_ids"]
    if not isinstance(rel, list) or not rel:
        raise ValueError(f"{where} must have a non-empty relevant_document_ids list")
    if not all(isinstance(x, str) for x in rel):
        raise ValueError(f"{where} relevant_document_ids must all be strings")

    chunks = q.get("relevant_chunk_ids")
    if chunks is not None and not isinstance(chunks, list):
        raise ValueError(f"{where} relevant_chunk_ids must be a list (or omitted)")
    return q


def load_golden_questions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden questions file must be a JSON list")
    return [validate_question(q, index=i) for i, q in enumerate(data)]
