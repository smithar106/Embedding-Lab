"""Central production configuration, read from environment variables.

All secrets and tunables live here. Non-secret values have sensible defaults.
Never commit real credentials — see ``.env.example`` (variable NAMES only).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load a local .env if present (no-op on Railway, where vars are injected).
# Does not override already-set environment variables.
load_dotenv()


@dataclass
class Settings:
    database_url: str = ""                 # required for DB-backed features
    deepseek_api_key: str = ""             # required for /ask
    generation_model: str = "deepseek-chat"
    embedding_model: str = "bge"
    top_k: int = 2
    log_level: str = "INFO"
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        # GENERATION_MODEL is the production name; DEEPSEEK_MODEL kept as an alias.
        generation_model=os.environ.get("GENERATION_MODEL")
        or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "bge"),
        top_k=int(os.environ.get("TOP_K", "2")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        cors_origins=os.environ.get("CORS_ORIGINS", "*"),
    )
