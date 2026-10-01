"""Verify the embedding model layer (loads models — cached after first run)."""
import numpy as np

from models.embedding_models import MODELS, embed_document, embed_query


def test_dimensions_match_metadata():
    for name, info in MODELS.items():
        vector = embed_query("a short test sentence", name)
        assert len(vector) == info.dim, f"{name} produced {len(vector)} dims, expected {info.dim}"


def test_dimensions_are_expected_values():
    assert MODELS["minilm"].dim == 384
    assert MODELS["bge"].dim == 768
    assert MODELS["e5"].dim == 768


def test_minilm_is_symmetric_no_prefix():
    # MiniLM uses no query/passage prefix, so query and document embeddings of
    # the same text are identical.
    q = np.asarray(embed_query("water on Mars", "minilm"))
    d = np.asarray(embed_document("water on Mars", "minilm"))
    assert np.array_equal(q, d)


def test_e5_uses_different_query_and_passage_prefixes():
    # E5 requires "query: " vs "passage: ", so the same text embeds differently.
    q = np.asarray(embed_query("water on Mars", "e5"))
    d = np.asarray(embed_document("water on Mars", "e5"))
    assert not np.array_equal(q, d)
