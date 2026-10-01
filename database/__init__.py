"""Database layer: PostgreSQL + pgvector for Embedding-Lab."""

# model_name -> the pgvector table holding that model's chunk embeddings.
# This mapping is the single source of truth for "which table does this model
# read/write", used by both the indexer and the retriever.
MODEL_TABLES = {
    "minilm": "chunk_embeddings_minilm",
    "bge": "chunk_embeddings_bge",
    "e5": "chunk_embeddings_e5",
}
