# Embedding-Lab

A small, transparent, educational project that loads **three Hugging Face
embedding models** and shows exactly how an embedding model becomes part of a
Python application — before any RAG system, database, or web UI is added on top.

| Key  | Model                                  | Dim  | Retrieval style |
|------|----------------------------------------|------|-----------------|
| `minilm` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | Symmetric (no prefixes) |
| `bge`    | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)                                 | 768 | Asymmetric (query instruction) |
| `e5`     | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)                                     | 768 | Asymmetric (`query:` / `passage:`) |

## Project layout

```
embedding-lab/
    README.md
    requirements.txt
    models/
        __init__.py          # re-exports the public interface
        embedding_models.py  # metadata + load + encode + cosine similarity
    scripts/
        test_embeddings.py   # one sentence through all three models
        compare_models.py    # rank passages by cosine similarity per model
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

The first run of each model downloads it from the Hugging Face Hub into the
local cache (default `~/.cache/huggingface/hub`). This happens once.

## Run

```bash
python scripts/test_embeddings.py     # dimensions, sample vector, latency
python scripts/compare_models.py      # which passage each model ranks first
```

## The interface

Two functions, used the same way for every model:

```python
from models.embedding_models import embed_document, embed_query

vec = embed_document("some passage of text", "bge")
q   = embed_query("a question", "bge")
```

The model-specific behaviour — required prefixes, dimensions — is written out
as plain data in `MODELS` inside `models/embedding_models.py`. Nothing is
hidden; open that file to see exactly what each model does.

## The execution path

```
        text
          │
          ▼
      tokenizer  ───────────  splits the string into sub-word tokens and maps
          │                   each token to an integer id from the model's
          │                   vocabulary ("token ids")
          ▼
      token IDs  ───────────  a list of integers, e.g. [2051, 2003, 1006, ...]
          │
          ▼
     transformer  ──────────  a stack of neural-network layers. It converts the
          │                   token ids into a vector per token, where each
          │                   token vector is *contextual* (depends on the
          │                   surrounding words)
          ▼
  contextual token
   representations
          │
          ▼
       pooling    ───────────  combines the per-token vectors into ONE vector
          │                   for the whole sentence (default: mean pooling)
          ▼
    embedding vector ────────  one fixed-length vector (384 or 768 floats)
          │
          ▼
   cosine similarity  ───────  cos θ = (a·b) / (|a|·|b|); 1.0 = same meaning
```

### What each step is

- **Tokenizer** — maps text → integer ids using the model's fixed vocabulary.
  Different models have different vocabularies, so the same sentence produces
  different token ids in different models.
- **Transformer weights** — the learned numbers (hundreds of millions of them)
  that were tuned during training. These are what make the model *do* something
  useful, and they are the bulk of the downloaded files.
- **`model.encode()`** — sentence-transformers' convenience method that runs
  the whole tokenizer → transformer → pooling pipeline for you in one call.
- **An embedding vector** — a point in a high-dimensional space where the model
  has learned to place similar-meaning text close together. It has no intrinsic
  "meaning" by itself; meaning lives in the *relative positions* of vectors.
- **Why different dimensions?** The dimension is just the width of the model's
  final output layer — an architectural choice, not a quality measure.
  MiniLM uses 384, BGE and E5 use 768.
- **Why query and document must use the same model?** Every model has its own
  coordinate system. A vector from MiniLM (384 dims) and a vector from E5
  (768 dims) cannot even be compared; and two 384-dim models still place text
  in *different* spaces, so mixing them silently produces garbage.
- **Why E5 uses `query:` / `passage:` prefixes?** E5 was trained with those
  role prefixes, so at inference time it expects them to know whether an input
  is a query or a passage. BGE similarly wants its search instruction on the
  query. Omitting the prefix is the classic "silent quality killer".
- **What cosine similarity measures?** The angle between two vectors,
  independent of their length. For embeddings, "how close to 1.0" ranks
  semantic similarity.

## What comes from us vs. from Hugging Face

**Our source code** (`models/embedding_models.py`, the scripts) contains:
- which model to load (its Hugging Face id),
- the metadata (dimension, family, prefixes),
- the tiny amount of glue (load once, encode, compute cosine),
- the demo scripts.

**The Hugging Face repository** (`sentence-transformers/all-MiniLM-L6-v2`,
`BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`) contains:
- `config.json` — the architecture description,
- the tokenizer files — the vocabulary and rules,
- the trained weights (`pytorch_model.bin` / `model.safetensors`) — the learned
  numbers.

We never write the model; we *download and run* it. Inspect the files yourself:

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://huggingface.co/intfloat/e5-base-v2
