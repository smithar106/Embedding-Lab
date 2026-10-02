"""One deterministic chunker (a recursive text splitter).

WHY chunk documents? Embedding models have a fixed input window (a few hundred
tokens) and produce ONE vector per input. A 20-page document cannot become one
useful vector — it would be a lossy average of everything. We split long
documents into overlapping chunks so each chunk is a focused, self-contained
unit of meaning, and each chunk gets its own vector. Retrieval then returns the
specific chunk that answers a query rather than a whole document.

WHY the SAME chunks for every model? The entire point of Embedding-Lab is to
compare how MiniLM / BGE / E5 retrieve differently. That comparison is only
meaningful if all three models see the IDENTICAL text units. If each model had
its own chunking, any difference in retrieval could come from the chunking, not
the model. So: chunk ONCE, embed three times.

The splitter is deterministic (same text -> same chunks) and configurable via
CHUNK_SIZE / CHUNK_OVERLAP.
"""
from __future__ import annotations

import hashlib

# Try these separators in order; each is "greedier" than the last. The final
# empty string splits on every character (last resort).
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Reasonable defaults (characters, not tokens — this is a simple educational
# splitter; a production system would use a real tokenizer).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text into segments no longer than ``chunk_size``."""
    if len(text) <= chunk_size:
        return [text] if text else []

    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        if sep == "":
            # Last resort: split into individual characters.
            pieces = list(text)
        else:
            pieces = text.split(sep)
            # Keep the separator attached so we never lose text (or meaning at
            # paragraph/sentence boundaries).
            pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]

        result: list[str] = []
        for piece in pieces:
            if piece.strip() == "":
                continue
            result.extend(_recursive_split(piece, separators[i:], chunk_size))
        return result

    # No separator matched (shouldn't happen) — hard-cut.
    return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]


def _merge(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge segments into ~chunk_size chunks carrying ``overlap`` from the
    previous chunk, so meaning that would otherwise be cut at the boundary is
    duplicated in both neighbours."""
    chunks: list[str] = []
    current = ""
    for seg in segments:
        candidate = (current + " " + seg).strip() if current else seg
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            # Overlap: keep the tail of the previous chunk.
            current = current[-overlap:] if overlap < len(current) else current
            current = (current + " " + seg).strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    document: dict,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split ONE document into identical chunks for all models.

    Returns a list of chunk dicts::

        {
          "chunk_id": "<document_id>#<chunk_index>",   # stable, deterministic
          "document_id": ...,
          "chunk_index": ...,
          "chunk_text": ...,
          "title": ..., "collection": ..., "topics": ...,
          "source_organization": ..., "source_url": ...,
          "metadata": ...,
          "content_hash": ...,   # sha256 of chunk_text (incremental upsert)
        }
    """
    text = document["text"]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    segments = _recursive_split(text, SEPARATORS, chunk_size)
    chunks_text = _merge(segments, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunks.append({
            "chunk_id": f"{document['document_id']}#{i}",
            "document_id": document["document_id"],
            "chunk_index": i,
            "chunk_text": chunk_text,
            "title": document["title"],
            "collection": document["collection"],
            "topics": document["topics"],
            "source_organization": document["source_organization"],
            "source_url": document["source_url"],
            "metadata": document["metadata"],
            "content_hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
        })
    return chunks
