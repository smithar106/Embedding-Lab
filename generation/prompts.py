"""Generation prompts + the citation format.

The prompt is DELIBERATELY constant across all three pipelines. The whole
point of Phase 4 is to isolate "how does changing the retrieved evidence change
the generated answer?" — so the generation model, prompt, temperature, and
retrieval settings must stay identical; only the evidence changes.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions using ONLY the supplied "
    "evidence. Follow these rules exactly:\n"
    "1. If the evidence does not contain enough information to answer the "
    "question, say clearly that the available evidence is insufficient.\n"
    "2. Do not introduce any factual claim that is not supported by the supplied "
    "evidence.\n"
    "3. Cite the evidence chunks you use inline, in square brackets, using the "
    "exact chunk id, for example [mars-liquid-water#0].\n"
    "4. Keep the answer concise and directly tied to the evidence."
)

# Chunk ids look like "<document_id>#<index>", e.g. "mars-liquid-water#0".
_CHUNK_CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+#\d+)\]")


def build_prompt(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Build the grounded prompt: QUESTION + RETRIEVED EVIDENCE (chunk id + text).

    This function does not depend on which embedding model produced the chunks —
    two runs with the same question and the same chunk texts yield the identical
    prompt, regardless of model.
    """
    blocks = [f"[{c['chunk_id']}]\n{c['chunk_text']}" for c in retrieved_chunks]
    evidence = "\n\n".join(blocks) if blocks else "(no evidence supplied)"

    user = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence}\n\n"
        "Answer the question using only the supplied evidence, citing chunks inline with [chunk_id]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_citations(answer: str | None) -> list[str]:
    """Extract cited chunk ids (``[chunk_id]``) from an answer, in order."""
    return _CHUNK_CITE_RE.findall(answer or "")
