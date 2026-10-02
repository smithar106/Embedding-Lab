# Deploying Embedding-Lab to Railway

A single FastAPI service + a single PostgreSQL/pgvector database. No frontend,
no MCP, no auth (this phase). The public endpoint should **not** be treated as a
high-volume unrestricted production service without authentication/rate
limiting.

## Architecture

```
        INTERNET
            |
         FastAPI                (api/app.py)
            |
          Agent                 (agent/agent.py)
       /          \
direct answer    tool call
                   |
            retrieval_search     (tools/retrieval_tool.py)
                   |
                  BGE             (loaded once at startup)
                   |
          PostgreSQL + pgvector
                   |
               Top-2 chunks
                   |
                 Agent
                   |
               DeepSeek
                   |
           grounded answer
```

## Deployment lifecycle

| When | What happens |
|------|--------------|
| **Once** | `python -m scripts.init_database` — create the pgvector extension + schema |
| **Per data update** | `python -m scripts.ingest_dataset` — chunk + embed + index documents |
| **Per deployment (app start)** | load BGE into memory, connect to the DB pool, become ready |
| **Per request** | question → agent decision → optional retrieval → generation → response |

Data ingestion is **separate** from the request path. The API never ingests.

## Railway setup steps

### A. Create the Railway project

In the Railway dashboard, create a new project from this GitHub repository.

### B. Add a PostgreSQL database with pgvector

Add a **PostgreSQL** service. Railway's PostgreSQL image includes `pgvector`; the
schema enables it with `CREATE EXTENSION IF NOT EXISTS vector`.

### C. Connect the GitHub repository

Connect `smithar106/Embedding-Lab`. Railway builds the `Dockerfile` and deploys.

### D. Configure variables

Set these service variables (names only — values live in Railway, not in git):

```
DATABASE_URL          (Railway auto-provides when you attach the Postgres service)
DEEPSEEK_API_KEY      (your DeepSeek key)
GENERATION_MODEL      = deepseek-chat
EMBEDDING_MODEL       = bge
TOP_K                 = 2
LOG_LEVEL             = INFO
HF_HOME               = /app/.hf_cache        (see "Model cache" below)
```

`CORS_ORIGINS` and `RATE_LIMIT_MAX` / `RATE_LIMIT_WINDOW` are optional.

### E. Model cache

On first deploy the app downloads `BAAI/bge-base-en-v1.5` (~440 MB) into
`HF_HOME`. To avoid re-downloading on every cold start, attach a Railway **volume**
mounted at `/app/.hf_cache`. Two distinct things happen:

- **Download** — the model files are fetched from the Hugging Face Hub into
  `HF_HOME` (disk). Happens once per cache location.
- **Load into memory** — at startup, the model is deserialized into process
  memory and reused for every request. Happens once per process.

Model weights are never committed to GitHub.

### F. Deploy

Railway builds the Dockerfile. The healthcheck polls `/health` (timeout 300s to
allow the first-run model download).

### G. Initialize the database (once)

Run the one-off init command (Railway → service → "Run command"):

```
python -m scripts.init_database
```

This enables `pgvector` and creates `documents`, `chunks`, and the three
`chunk_embeddings_*` tables, then verifies them.

### H. Run ingestion (per data update)

```
python -m scripts.ingest_dataset --path data/raw/hard
```

This chunks the documents once and embeds them with every configured model,
storing the vectors in pgvector.

### I–L. Test

```
GET  /health      -> {"status":"healthy","database":"connected","embedding_model":"loaded"}
POST /retrieve    -> {"query":"...","results":[...]}
POST /ask         -> {"answer":"...","citations":[...],"tool_used":true,...}
GET  /config      -> {"generation_model":...,"embedding_model":...,"top_k":...}
```

## Local run (before Railway)

```bash
python -m scripts.init_database
python -m scripts.ingest_dataset --path data/raw/hard
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
