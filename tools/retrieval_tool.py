"""retrieval_search() — the retrieval capability behind one clean tool.

This is an ABSTRACTION BOUNDARY. The agent (and anything above this layer) sees
only ``retrieval_search(query, top_k, ...)`` and its description. It must NOT
know about pgvector, HNSW, embedding dimensions, embedding prefixes, SQL, vector
tables, chunking, indexing, or cosine similarity — all of that is implemented
below this function and kept out of sight.

The tool returns EVIDENCE, not a final answer. Deciding what to do with the
evidence — and synthesizing an answer — is the agent's job, not the tool's.
"""
from __future__ import annotations

import os

from retrieval.retriever import retrieve

# Experimentally selected defaults (see the README / Phase 4.5):
#   - BGE had stronger Recall@1 than MiniLM, with equivalent downstream fact
#     coverage by K=2.
#   - K=2 reached the observed quality plateau; K>2 added context without
#     measurable answer-quality improvement.
# These are a choice from THIS experiment — not a claim of universal optimality.
DEFAULT_EMBEDDING_MODEL = "bge"
DEFAULT_TOP_K = 2

TOOL_DESCRIPTION = (
    "Search the knowledge base for evidence relevant to the user's question. "
    "Use this tool when the question requires information contained in the "
    "knowledge base."
)


def retrieval_search(
    query: str,
    top_k: int | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Return ranked evidence chunks for ``query`` from the knowledge base.

    This calls the EXISTING retrieval system (``retrieve``) — no retrieval logic
    is duplicated here. Returns structured data, with raw implementation details
    (embeddings, SQL, vector search) kept internal.

    ``top_k`` and ``embedding_model`` default to the environment configuration
    (``TOP_K`` / ``EMBEDDING_MODEL``), so a production deployment can pin them
    without code changes.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k is None:
        top_k = int(os.environ.get("TOP_K", DEFAULT_TOP_K))
    if embedding_model is None:
        embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    results = retrieve(query, embedding_model, top_k)
    return {
        "query": query,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "title": r["title"],
                "text": r["chunk_text"],
                "rank": r["rank"],
            }
            for r in results
        ],
    }


# The tool schema the LLM sees for function/tool calling. Only query + top_k are
# exposed; the embedding model stays at its (experimentally selected) default so
# the agent doesn't need to know it exists.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieval_search",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased to find relevant evidence.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of evidence chunks to return (default 2).",
                },
            },
            "required": ["query"],
        },
    },
}
