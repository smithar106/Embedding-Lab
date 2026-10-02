"""
Embedding models — a thin, explicit wrapper over sentence-transformers.

This file is deliberately small and transparent. Its whole job is to answer one
question: "how does an embedding model become part of a Python application?"

The answer, in four steps, is:

    1.  You name the model (its Hugging Face repository id, e.g.
        "BAAI/bge-base-en-v1.5").
    2.  ``SentenceTransformer(hf_id)`` downloads the model files and builds the
        neural network in memory (see ``_model`` below).
    3.  You call ``model.encode(text)`` to turn a string into a vector (see
        ``embed_document`` / ``embed_query``).
    4.  You compare vectors with cosine similarity (see ``cosine_similarity``).

The three models compared here are:

    - "minilm"  -> sentence-transformers/all-MiniLM-L6-v2   (384-dim, symmetric)
    - "bge"     -> BAAI/bge-base-en-v1.5                    (768-dim, retrieval)
    - "e5"      -> intfloat/e5-base-v2                      (768-dim, retrieval)

Only the model-specific differences are encoded here, and they are encoded as
plain data (a dataclass per model) — nothing is hidden behind clever
abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # typing only — the real import happens lazily inside _model()
    from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. Model metadata (plain data — the whole "model vocabulary" lives here)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    """Static facts about one embedding model.

    ``query_prefix`` / ``document_prefix`` are the one real behavioural
    difference between models: some models were trained to expect a role word
    (or instruction) prepended to the input, and feeding them text without it
    measurably degrades quality. We store the required prefix per model so the
    encode functions can apply it automatically.
    """

    key: str            # short name used in code: "minilm", "bge", "e5"
    hf_id: str          # Hugging Face repository id (what you pass to load it)
    dim: int            # output vector dimension (width of the final layer)
    family: str         # who makes it / which family it belongs to
    use: str            # the retrieval use it was trained/optimised for
    hf_url: str         # link to inspect the model files on the Hugging Face Hub
    query_prefix: str = ""     # prepended to a retrieval QUERY   (if required)
    document_prefix: str = ""  # prepended to a DOCUMENT/passage  (if required)


MODELS: dict[str, ModelInfo] = {
    "minilm": ModelInfo(
        key="minilm",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        family="MiniLM (sentence-transformers)",
        use="Symmetric semantic similarity / general sentence embeddings",
        hf_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        # MiniLM is a SYMMETRIC model: queries and documents are encoded the
        # same way, with no role prefix. Good default for paraphrase/similarity.
        query_prefix="",
        document_prefix="",
    ),
    "bge": ModelInfo(
        key="bge",
        hf_id="BAAI/bge-base-en-v1.5",
        dim=768,
        family="BGE — BAAI General Embedding",
        use="Asymmetric retrieval (short query -> long passage)",
        hf_url="https://huggingface.co/BAAI/bge-base-en-v1.5",
        # BGE recommends prepending an *instruction* to the QUERY side only.
        # Documents/passages are encoded with no prefix.
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
    ),
    "e5": ModelInfo(
        key="e5",
        hf_id="intfloat/e5-base-v2",
        dim=768,
        family="E5 — intfloat (EmbEddings from bidirEctional Encoder rEpresentations)",
        use="Asymmetric retrieval (query -> passage)",
        hf_url="https://huggingface.co/intfloat/e5-base-v2",
        # E5 REQUIRES a role prefix on every input: "query: " for queries and
        # "passage: " for documents. Omitting the prefixes is a well-known way
        # to silently tank its retrieval quality, so we always apply them.
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


# ---------------------------------------------------------------------------
# 2. Loading a model (and where the files actually live on disk)
# ---------------------------------------------------------------------------
# When you call SentenceTransformer("BAAI/bge-base-en-v1.5") for the first time:
#
#   1. The Hugging Face Hub library downloads the model's files — config.json,
#      tokenizer files, and the trained weights (pytorch_model.bin or
#      model.safetensors) — into a local cache, by default
#      ``~/.cache/huggingface/hub`` (overridable with the HF_HOME env var).
#   2. It reads config.json, learns the architecture (e.g. a 6- or 12-layer
#      BERT-style transformer), and constructs that network in memory with
#      random weights.
#   3. It copies the downloaded (learned) weights into the network.
#
# The download happens ONCE. Every later run reuses the cache, so only the very
# first run of each model is slow (it fetches a few hundred MB).

_loaded: dict[str, "SentenceTransformer"] = {}


def _model(model_name: str) -> "SentenceTransformer":
    """Return the loaded model, downloading/loading it once and caching it."""
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Choose from {list(MODELS)}")

    if model_name not in _loaded:
        # Imported lazily: sentence-transformers pulls in PyTorch, which is
        # slow to import, so we only pay that cost when we actually load a model.
        from sentence_transformers import SentenceTransformer

        _loaded[model_name] = SentenceTransformer(MODELS[model_name].hf_id)
    return _loaded[model_name]


def is_model_loaded(model_name: str) -> bool:
    """Whether ``model_name`` has been loaded into process memory (cached)."""
    return model_name in _loaded


# ---------------------------------------------------------------------------
# 3. Encoding text (the model-agnostic interface)
# ---------------------------------------------------------------------------
# What model.encode(text) actually does, end to end:
#
#   text
#     -> tokenizer  : splits the string into sub-word tokens and maps each one
#                     to an integer id ("token ids") from the model's vocabulary
#     -> transformer : runs the token ids through the stacked neural network
#                     layers; each token becomes a *contextual* vector (a token
#                     only means something in the context of the sentence)
#     -> pooling     : combines the per-token vectors into ONE vector for the
#                     whole sentence (sentence-transformers default is mean
#                     pooling: average the token vectors)
#     -> embedding   : that single vector, optionally L2-normalised
#
# We normalise (unit length) so that cosine similarity reduces to a dot product
# and all comparisons live on the same scale.


def embed_document(text: str, model_name: str):
    """Encode a document/passage with the named model.

    Only E5 prepends anything here ("passage: "); MiniLM and BGE encode the
    passage as-is.
    """
    info = MODELS[model_name]
    model = _model(model_name)

    input_text = (info.document_prefix + text) if info.document_prefix else text
    return model.encode(input_text, normalize_embeddings=True)


def embed_query(text: str, model_name: str):
    """Encode a retrieval query with the named model.

    BGE prepends its search instruction and E5 prepends "query: "; MiniLM uses
    the raw text. Applying the right prefix is what keeps retrieval quality up.
    """
    info = MODELS[model_name]
    model = _model(model_name)

    input_text = (info.query_prefix + text) if info.query_prefix else text
    return model.encode(input_text, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# 4. Comparing vectors
# ---------------------------------------------------------------------------
def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors: ``cos θ = (a·b) / (|a|·|b|)``.

    Returns a value in [-1, 1]:

        *  1.0 -> the two vectors point the same way (near-identical meaning)
        *  0.0 -> orthogonal (no shared direction; unrelated)
        * -1.0 -> opposite direction

    For embeddings we only care about "how close to 1.0" — that ranks semantic
    closeness. Note the denominator normalises for vector length, so cosine
    measures *direction* (meaning) rather than magnitude.
    """
    import numpy as np  # numpy comes as a dependency of sentence-transformers

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
