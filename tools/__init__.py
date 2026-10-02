"""Tool layer — retrieval exposed as ONE clean, callable tool."""
from tools.retrieval_tool import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    TOOL_DESCRIPTION,
    TOOL_SCHEMA,
    retrieval_search,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_TOP_K",
    "TOOL_DESCRIPTION",
    "TOOL_SCHEMA",
    "retrieval_search",
]
