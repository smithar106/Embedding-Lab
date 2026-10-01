"""Embedding-Lab — a small, transparent tour of Hugging Face embedding models.

This package exposes a single, model-agnostic interface::

    from models.embedding_models import embed_document, embed_query

    vec = embed_document("some passage", "bge")
    q   = embed_query("a question", "bge")

Everything model-specific (Hugging Face id, dimension, required prefixes) is
written out as plain data in :mod:`models.embedding_models` — deliberately, so
you can see exactly how each model behaves.
"""

from .embedding_models import (
    MODELS,
    ModelInfo,
    cosine_similarity,
    embed_document,
    embed_query,
)

__all__ = [
    "MODELS",
    "ModelInfo",
    "cosine_similarity",
    "embed_document",
    "embed_query",
]
