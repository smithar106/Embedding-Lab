# Embedding-Lab

A small, transparent, educational project for understanding how embedding
models become part of an application — first the **embedding layer** (Phase 1),
then the **indexing + retrieval layer** (Phase 2). It loads three Hugging Face
models, chunks a dataset **once**, embeds every chunk with all three models,
stores the vectors in PostgreSQL + pgvector, and lets you retrieve and compare
the results per model.

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
    .env.example
    models/
        __init__.py          # Phase 1: the common embed_document / embed_query
        embedding_models.py  #        metadata, loading, encoding, cosine
    database/
        __init__.py          # Phase 2: model_name -> pgvector table mapping
        connection.py        #        psycopg connection + schema bootstrap
        schema.sql           #        documents / chunks / 3 embedding tables
    ingestion/
        __init__.py
        loader.py            # .txt / .json / .jsonl -> normalized documents
        chunker.py           # one recursive text splitter (chunk ONCE)
        indexer.py           # load -> chunk -> embed(3 models) -> store
    retrieval/
        __init__.py
        retriever.py         # retrieve(query, model_name, top_k)
        comparison.py        # retrieve all models + overlap
    evaluation/
        __init__.py
        dataset.py           # load + validate golden questions
        metrics.py           # Recall@K, MRR, Hit Rate — readable Python
        evaluator.py         # run questions through every model, aggregate
    data/
        raw/sample/          # tiny TEST documents (clearly labelled)
        raw/hard/            # synthetic harder corpus (lexical vs semantic)
        evaluation/          # golden_questions.json / golden_questions_hard.json
        processed/
    scripts/
        ingest_dataset.py
        test_retrieval.py
        compare_retrieval.py
        evaluate_models.py   # Phase 3: golden-question evaluation + benchmark
    results/                 # saved evaluation runs (gitignored)
    tests/
```

## Setup

### 1. Python

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

The first run of each model downloads it from the Hugging Face Hub into the
local cache (default `~/.cache/huggingface/hub`). This happens once.

### 2. PostgreSQL + pgvector

You need a running PostgreSQL with the `pgvector` extension:

```bash
# Homebrew (macOS)
brew install pgvector          # or build against a specific PG with pg_config

createdb embedding_lab
psql -d embedding_lab -c "CREATE EXTENSION vector;"
```

Copy `.env.example` to `.env` if your connection string is not the local
default (`postgresql://localhost:5432/embedding_lab`).

## Run

```bash
# Phase 1 — embeddings only (no database)
python scripts/test_embeddings.py
python scripts/compare_models.py

# Phase 2 — index the sample dataset, then retrieve/compare
python scripts/ingest_dataset.py          # load -> chunk -> embed -> pgvector
python scripts/test_retrieval.py          # one query, top-5 per model
python scripts/compare_retrieval.py       # rankings + overlap + latency

# Phase 3 — evaluate retrieval against ground truth
python scripts/evaluate_models.py         # golden questions -> Recall@K / MRR
# harder corpus (semantic vs lexical):
python scripts/ingest_dataset.py --reset --path data/raw/hard
python scripts/evaluate_models.py --questions data/evaluation/golden_questions_hard.json
```

## Architecture

**Indexing time** (the one-off, expensive work):

```
                     DATASET
                        |
                     loader
                        |
                     chunker
                        |
                IDENTICAL CHUNKS
                        |
          +-------------+-------------+
          |             |             |
       MiniLM          BGE            E5
        384d           768d          768d
          |             |             |
          +-------------+-------------+
                        |
                 PostgreSQL
                   pgvector
                        |
                 retrieval layer
```

**Query time** (cheap; the model is already loaded, vectors already stored):

```
                       QUERY
                         |
          +--------------+--------------+
          |              |              |
       MiniLM           BGE            E5
          |              |              |
      query vector   query vector   query vector
          |              |              |
      vector search  vector search  vector search
          |              |              |
        Top-K          Top-K          Top-K
          +--------------+--------------+
                         |
                  compare results
```

## The interface

Two functions, used the same way for every model (Phase 1):

```python
from models.embedding_models import embed_document, embed_query

vec = embed_document("some passage of text", "bge")
q   = embed_query("a question", "bge")
```

And the retrieval interface (Phase 2):

```python
from retrieval.retriever import retrieve

results = retrieve("What evidence is there that Mars once had water?", "e5", top_k=5)
```

Model-specific behaviour (prefixes, dimensions) is written out as plain data in
`MODELS` — open `models/embedding_models.py` to see exactly what each model does.

## Schema decision — one chunk, three embeddings

The chunk text is stored **once** in `chunks`. Each model's vectors live in its
own table:

- `chunk_embeddings_minilm` — `vector(384)`
- `chunk_embeddings_bge` — `vector(768)`
- `chunk_embeddings_e5` — `vector(768)`

Why three tables instead of one `chunk_embeddings(model_name, embedding)`?
Because **pgvector requires a fixed dimension** for a `vector` column (and for
the HNSW index), and MiniLM is 384-dim while BGE/E5 are 768-dim. A single
`vector` column cannot cleanly hold both with a usable index. Three
dimension-specific tables is the simplest technically correct design, and it
mirrors the rule that **one logical chunk → three embedding representations**.

## Incremental indexing (content hashing)

Each chunk carries a `content_hash` (sha256 of its text). On re-ingestion:

- a chunk whose hash matches what's already stored is **not re-embedded**;
- a new or changed chunk is embedded (with all three models) and upserted;
- chunks that no longer exist are deleted (their embeddings cascade away).

This is the hook for dynamic datasets: later, only *changed* documents re-run
the expensive embedding step.

## Retrieval

`retrieve(query, model_name, top_k)` embeds the query with that model's own
query-side handling, then runs a cosine-similarity search **only** against that
model's table:

```sql
SELECT ..., 1 - (e.embedding <=> %s) AS similarity
FROM chunk_embeddings_e5 e JOIN chunks c ON c.chunk_id = e.chunk_id
ORDER BY e.embedding <=> %s
LIMIT 5
```

(`<=>` is pgvector's cosine *distance*; `1 - distance` = similarity.)

### Why embedding spaces cannot be mixed

Every model has its own coordinate system. A MiniLM query is a point in a
384-dimensional space; BGE and E5 are 768-dimensional, and even the two 768-dim
models are different spaces. Comparing a MiniLM query against BGE embeddings is
mathematically meaningless (and dimensionally impossible for 384 vs 768). The
`MODEL_TABLES` mapping enforces that each query is searched only in its own
space.

### Why raw similarity scores are NOT comparable across models

`E5 = 0.85` vs `MiniLM = 0.61` does **not** mean E5 is better. Each model has a
different similarity-score distribution. Compare **rankings**, **retrieved
chunks**, **overlap**, and **latency** — not raw scores. (Ground-truth metrics
like Recall@K, MRR, and nDCG will be added in a later phase.)

## The execution path

```
        text
          │
          ▼
      tokenizer  ───────────  splits text into sub-word tokens, maps each to an
          │                   integer id ("token ids") from the model vocab
          ▼
      token IDs  ───────────  e.g. [2051, 2003, 1006, ...]
          │
          ▼
     transformer  ──────────  stacked neural-network layers turn each token id
          │                   into a *contextual* vector (meaning depends on
          │                   surrounding words)
          ▼
       pooling    ───────────  combines per-token vectors into ONE vector for
          │                   the whole sentence (default: mean pooling)
          ▼
    embedding vector ────────  one fixed-length vector (384 or 768 floats)
          │
          ▼
   cosine similarity  ───────  cos θ = (a·b) / (|a|·|b|); 1.0 = same meaning
```

### Key concepts

- **Tokenizer** — text → integer ids, using the model's fixed vocabulary.
- **Transformer weights** — the learned numbers (hundreds of millions) that were
  tuned during training; the bulk of the downloaded files.
- **`model.encode()`** — runs tokenizer → transformer → pooling in one call.
- **An embedding vector** — a point in high-dimensional space; meaning lives in
  the *relative* positions of vectors, not any single number.
- **Why dimensions differ** — it's just the width of the model's final output
  layer (an architectural choice), not a quality measure.
- **Why query + document must use the same model** — different models place text
  in different spaces; mixing them silently produces garbage.
- **Why E5/BGE use prefixes** — they were trained with those role words and
  expect them at inference; omitting them is the classic silent quality killer.
- **Why chunk before embedding** — models have a small input window and return
  one vector per input; a long document must be split so each focused chunk
  gets its own embedding, and retrieval returns the *specific* chunk that
  answers a query.
- **Why identical chunks across models** — so any difference in retrieval comes
  from the *model*, not from different chunking.

## Indexing time vs query time

- **Indexing time**: documents → chunks → document embeddings → pgvector. Heavy
  (embeds every chunk with every model), run once / incrementally.
- **Query time**: question → query embedding → vector similarity search → Top-K
  chunks. Light (one embedding + one indexed search).

## From similarity search to retrieval evaluation

Phase 2 answered *"what does the model retrieve?"*; Phase 3 answers *"is what it
retrieves actually correct?"*.

```
Phase 2:   Question -> embedding -> similarity search -> Top-K chunks

Phase 3:   Question -> retrieval -> Top-K chunks
                        -> compare against known relevant evidence
                        -> compute retrieval metrics (Recall@K, MRR, ...)
```

Why the distinction matters: a model producing plausible-looking results is not
enough. Two models can return completely different chunks and both *look*
reasonable. Only ground truth — knowing which documents actually contain the
answer — lets us measure retrieval quality objectively.

### Ground truth

`data/evaluation/golden_questions.json` maps each question to the document IDs
(and optionally chunk IDs) that contain the answer. Ground truth is authored by
hand against our source documents — no model decides relevance during
evaluation.

- **Document-level relevance** — "did the system retrieve evidence from the
  correct document?" (default).
- **Chunk-level relevance** — "did the system retrieve the exact relevant
  passage?" (optional `relevant_chunk_ids`, for fine-grained evaluation later).

### Metrics

- **Recall@K** — fraction of relevant items retrieved within the top-K results
  (`|top-k ∩ relevant| / |relevant|`).
- **Hit Rate@K** — whether *any* relevant item appears in the top-K (binary).
- **MRR (Mean Reciprocal Rank)** — the mean of `1 / rank_of_first_relevant`
  across questions.

These compare **rankings and retrieval effectiveness**, never raw cosine scores.
`E5 = 0.85` vs `MiniLM = 0.62` is not a comparison — each model has its own score
distribution. We compare Recall@K, MRR, rank position, ground-truth relevance,
and latency.

### What the benchmark does and does not tell us

A benchmark on *this* tiny corpus tells us how the three models rank the
documents *in this dataset*. It does **not** establish a universal "best" model:
results depend on the dataset, query distribution, ground-truth labels, chunking
strategy, and retrieval settings (top-k, similarity metric). A later phase with
a larger real dataset is what turns this into a meaningful comparison.

## What comes from us vs. from Hugging Face

**Our source code** (loader, chunker, indexer, retriever, scripts) contains the
data shape, the chunking rule, the database schema, the retrieval query, and the
thin glue to load/encode/compare.

**The Hugging Face repositories** contain `config.json` (architecture), the
tokenizer files (vocabulary), and the trained weights (`pytorch_model.bin` /
`model.safetensors`). We download and run them — we never write them:

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://huggingface.co/intfloat/e5-base-v2

## Tests

```bash
python -m pytest -q
```

Covers: embedding dimensions (384/768/768), symmetric vs prefixed models,
deterministic chunking, loader normalization, schema dimensions, identical chunk
IDs across models, top-k count, idempotent re-ingestion, and the retrieval
metrics (Recall@K, RR, MRR, Hit Rate) plus golden-dataset validation and
evaluator serialization. The database tests skip automatically when
PostgreSQL/pgvector is not reachable.
