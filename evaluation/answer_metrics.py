"""Answer-level evaluation metrics (transparent, no LLM-evaluation framework).

Retrieval quality and answer quality are DIFFERENT problems. Retrieval metrics
(Recall@K, MRR) measure whether the right chunks were found; these metrics
measure whether the generated answer is grounded in — and cites — the evidence
that was actually supplied.
"""
from __future__ import annotations

from generation.prompts import parse_citations


def fact_coverage(answer: str | None, expected_facts: list[str]) -> dict:
    """How many expected facts appear verbatim (case-insensitive) in the answer.

    Human-authored, deterministic ground truth — we do not ask a model to judge.
    """
    if not expected_facts:
        return {"matched": 0, "total": 0, "coverage": None, "matched_facts": [], "missing_facts": []}
    text = (answer or "").lower()
    matched = [f for f in expected_facts if f.lower() in text]
    missing = [f for f in expected_facts if f.lower() not in text]
    return {
        "matched": len(matched),
        "total": len(expected_facts),
        "coverage": len(matched) / len(expected_facts),
        "matched_facts": matched,
        "missing_facts": missing,
    }


def citation_validity(citations: list[str], supplied_chunk_ids: list[str]) -> dict:
    """Every citation must correspond to a chunk actually supplied to the model.

    A citation that names a chunk that was NOT in the retrieved context is a
    hallucinated citation. ``ratio`` = valid / total (1.0 when there are no
    citations at all, since there is nothing invalid to flag).
    """
    supplied = set(supplied_chunk_ids)
    valid = [c for c in citations if c in supplied]
    invalid = [c for c in citations if c not in supplied]
    total = len(citations)
    return {
        "valid": len(valid),
        "invalid": len(invalid),
        "total": total,
        "ratio": (len(valid) / total) if total else 1.0,
        "invalid_citations": invalid,
    }


def citation_relevance(citations: list[str], relevant_document_ids: list[str]) -> dict:
    """Whether cited chunks belong to a ground-truth relevant document.

    A chunk id ``doc#i`` is relevant if ``doc`` is in ``relevant_document_ids``.
    """
    rel = set(relevant_document_ids)
    relevant = [c for c in citations if c.split("#")[0] in rel]
    total = len(citations)
    return {
        "relevant": len(relevant),
        "total": total,
        "ratio": (len(relevant) / total) if total else None,
        "relevant_citations": relevant,
    }


def is_refusal(answer: str | None) -> bool:
    """Whether the answer abstains (insufficient-evidence behaviour)."""
    text = (answer or "").lower()
    markers = (
        "insufficient", "not enough", "cannot answer", "do not have enough",
        "no evidence", "does not contain", "not supported", "cannot determine",
    )
    return any(m in text for m in markers)
