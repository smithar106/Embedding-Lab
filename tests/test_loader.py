"""Verify the document loader normalizes .txt and .json into the same shape."""
import json

from ingestion.loader import load_documents


def test_load_txt():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "doc.txt"
        f.write_text("My Title\nThis is the body text.")
        docs = load_documents(f)
    assert len(docs) == 1
    assert docs[0]["title"] == "My Title"
    assert "body text" in docs[0]["text"]
    assert docs[0]["source_organization"] == "doc.txt"


def test_load_json_list():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "docs.json"
        f.write_text(json.dumps([
            {"document_id": "a", "title": "A", "text": "hello", "source": "s"},
            {"document_id": "b", "title": "B", "text": "world"},
        ]))
        docs = load_documents(f)
    assert [x["document_id"] for x in docs] == ["a", "b"]
    assert docs[1]["source_organization"] == "docs.json"  # default source = filename


def test_load_json_wrapped():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "wrapped.json"
        f.write_text(json.dumps({"documents": [{"text": "only text here"}]}))
        docs = load_documents(f)
    assert len(docs) == 1
    assert "document_id" in docs[0]  # auto-generated


def test_load_directory_mixed_formats():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        (p / "a.txt").write_text("Title A\nbody a")
        (p / "b.txt").write_text("Title B\nbody b")
        (p / "c.json").write_text(json.dumps([{"title": "C", "text": "body c"}]))
        docs = load_documents(p)
    assert len(docs) == 3
