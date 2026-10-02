"""Verify the chunker produces deterministic, model-agnostic chunks."""
from ingestion.chunker import chunk_document


def _doc(text, doc_id="d1"):
    return {
        "document_id": doc_id, "title": "t", "text": text, "summary": "",
        "collection": "uncategorized", "topics": [],
        "source_organization": "test", "source_url": None, "metadata": {},
    }


def test_chunking_is_deterministic():
    text = "word " * 4000
    a = chunk_document(_doc(text), chunk_size=400, chunk_overlap=50)
    b = chunk_document(_doc(text), chunk_size=400, chunk_overlap=50)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert [c["chunk_text"] for c in a] == [c["chunk_text"] for c in b]


def test_chunk_fields_present():
    chunks = chunk_document(_doc("word " * 2000))
    assert len(chunks) > 1
    required = {"chunk_id", "document_id", "chunk_index", "chunk_text", "title", "collection", "topics", "source_organization", "source_url", "metadata", "content_hash"}
    for c in chunks:
        assert required <= set(c.keys())


def test_chunk_ids_are_stable_and_sequential():
    chunks = chunk_document(_doc("word " * 2000))
    assert [c["chunk_id"] for c in chunks] == [f"d1#{i}" for i in range(len(chunks))]
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_short_text_is_single_chunk():
    assert len(chunk_document(_doc("a short sentence."))) == 1


def test_overlap_preserves_tail():
    # With overlap, the start of chunk 2 reuses the end of chunk 1.
    text = "word " * 3000
    chunks = chunk_document(_doc(text), chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= 2
    # The last words of chunk 0 should appear at the start of chunk 1.
    tail = chunks[0]["chunk_text"].split()[-10:]
    head = chunks[1]["chunk_text"].split()[:10]
    assert tail == head
