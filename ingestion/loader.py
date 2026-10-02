"""Generic document loader.

Every source of documents is normalised to ONE shape::

    {
        "document_id": "...",          # stable unique id (primary key)
        "title": "...",
        "text": "...",                 # the full original text
        "summary": "...",              # one-paragraph abstract (may be empty)
        "collection": "climate",       # one of the library's collections
        "topics": ["emissions", ...],  # free-form tags
        "source_organization": "...",  # e.g. "International Energy Agency"
        "source_url": "https://...",   # canonical URL of the source
        "publication_date": "2025-11-12",  # ISO date or None
        "retrieved_at": "2026-10-01T...",  # ISO timestamp or None (fetch time)
        "license": "CC BY 4.0",        # reuse license (required for the library)
        "metadata": {...},             # any extra fields, free-form
    }

This module understands three formats:

    * ``.txt``   — a plain-text file; the first non-empty line is the title and
                   the rest is the body.
    * ``.json``  — either a single document object, a list of document objects,
                   or ``{"documents": [...]}``.
    * ``.jsonl`` — one JSON document object per line (the source-catalog/corpus
                   format).

To add a new format later, add a loader function and register it in the
``_LOADERS`` dict below keyed by file extension — the rest of the pipeline does
not need to change.
"""
from __future__ import annotations

import json
from pathlib import Path

# extension -> loader function that returns a list of normalized document dicts
_LOADERS = {}

COLLECTIONS = (
    "climate",
    "energy",
    "food-agriculture",
    "water",
    "cities",
    "global-development",
)


def _register(ext: str):
    def decorator(fn):
        _LOADERS[ext] = fn
        return fn
    return decorator


def _normalize(raw: dict, *, default_source: str, fallback_id: int) -> dict:
    """Coerce a raw dict into the canonical document shape, filling defaults."""
    text = raw.get("text") or raw.get("content") or ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"document has no 'text' (source: {default_source})")

    document_id = raw.get("document_id") or raw.get("id") or f"{default_source}-{fallback_id}"
    title = raw.get("title") or raw.get("name") or document_id
    collection = str(raw.get("collection") or "uncategorized")
    topics = raw.get("topics") or []
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(",") if t.strip()]
    source_organization = str(raw.get("source_organization") or raw.get("source") or default_source)
    source_url = raw.get("source_url")
    publication_date = raw.get("publication_date")
    retrieved_at = raw.get("retrieved_at")
    license_ = str(raw.get("license") or "unknown")
    summary = str(raw.get("summary") or "")
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    return {
        "document_id": str(document_id),
        "title": str(title),
        "text": text.strip(),
        "summary": summary,
        "collection": collection,
        "topics": [str(t) for t in topics],
        "source_organization": source_organization,
        "source_url": source_url,
        "publication_date": publication_date,
        "retrieved_at": retrieved_at,
        "license": license_,
        "metadata": metadata,
    }


@_register(".txt")
def _load_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else path.stem
    body = "\n".join(lines[1:]) if len(lines) > 1 else text
    return [{
        "document_id": path.stem,
        "title": title,
        "text": body or text,
        "summary": "",
        "collection": "uncategorized",
        "topics": [],
        "source_organization": path.name,
        "source_url": None,
        "publication_date": None,
        "retrieved_at": None,
        "license": "unknown",
        "metadata": {"format": "txt"},
    }]


@_register(".json")
def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "documents" in data:
        items = data["documents"]
    elif isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"unsupported JSON shape in {path.name}")
    return [_normalize(d, default_source=path.name, fallback_id=i) for i, d in enumerate(items)]


@_register(".jsonl")
def _load_jsonl(path: Path) -> list[dict]:
    docs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        docs.append(_normalize(json.loads(line), default_source=path.name, fallback_id=i))
    return docs


def load_documents(path: str | Path) -> list[dict]:
    """Load every document under a file or directory into normalized dicts."""
    path = Path(path)
    docs: list[dict] = []

    if path.is_file():
        ext = path.suffix.lower()
        loader = _LOADERS.get(ext)
        if loader is None:
            raise ValueError(f"unsupported file type: {ext}")
        docs.extend(loader(path))
    elif path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() in _LOADERS:
                docs.extend(_LOADERS[file.suffix.lower()](file))
    else:
        raise FileNotFoundError(f"no such file or directory: {path}")

    # Guard against duplicate document ids across files (would break upserts).
    seen = set()
    for d in docs:
        if d["document_id"] in seen:
            d["document_id"] = f"{d['document_id']}--{len(seen)}"
        seen.add(d["document_id"])

    return docs
