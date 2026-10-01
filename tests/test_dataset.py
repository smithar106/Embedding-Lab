"""Unit tests for golden-question dataset loading + validation."""
import json
import tempfile
from pathlib import Path

import pytest

from evaluation.dataset import load_golden_questions, validate_question


def _write(data) -> Path:
    tmp = tempfile.TemporaryDirectory()
    f = Path(tmp.name) / "g.json"
    f.write_text(json.dumps(data))
    # keep the temp dir alive until the test is done by returning a closure-friendly path;
    # simpler: write to a module-level temp file. For brevity use a fixed temp file.
    return f, tmp


def test_load_valid_with_optional_chunk_ids():
    tmp = tempfile.TemporaryDirectory()
    f = Path(tmp.name) / "g.json"
    f.write_text(json.dumps([
        {"question_id": "q1", "question": "q?", "relevant_document_ids": ["a"]},
        {"question_id": "q2", "question": "q2?", "relevant_document_ids": ["b", "c"],
         "relevant_chunk_ids": ["b#0"]},
    ]))
    qs = load_golden_questions(f)
    assert len(qs) == 2
    assert qs[1]["relevant_chunk_ids"] == ["b#0"]


def test_validate_missing_required_key():
    with pytest.raises(ValueError):
        validate_question({"question": "no ids"})


def test_validate_empty_relevant():
    with pytest.raises(ValueError):
        validate_question({"question_id": "q", "question": "x", "relevant_document_ids": []})


def test_validate_relevant_not_strings():
    with pytest.raises(ValueError):
        validate_question({"question_id": "q", "question": "x", "relevant_document_ids": [1, 2]})


def test_load_non_list_raises():
    tmp = tempfile.TemporaryDirectory()
    f = Path(tmp.name) / "g.json"
    f.write_text(json.dumps({"documents": []}))
    with pytest.raises(ValueError):
        load_golden_questions(f)
