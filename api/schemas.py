"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field

MAX_QUESTION_LENGTH = 2000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=50)


class LatencyMs(BaseModel):
    agent_decision: float = 0.0
    retrieval: float = 0.0
    generation: float = 0.0
    total: float = 0.0


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    tool_used: bool = False
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: LatencyMs
    token_usage: dict | None = None
    request_id: str


class RetrieveResponse(BaseModel):
    query: str
    embedding_model: str
    top_k: int
    results: list[dict] = Field(default_factory=list)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=5)


class CompareResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str
    text: str
    similarity_score: float


class CompareGroup(BaseModel):
    model_key: str
    model_name: str
    results: list[CompareResult] = Field(default_factory=list)


class CompareResponse(BaseModel):
    query: str
    top_k: int
    groups: list[CompareGroup] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    generation_model: str
    embedding_model: str
    top_k: int


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str


class CollectionStat(BaseModel):
    id: str
    name: str
    documents: int
    passages: int
    topics: list[str] = Field(default_factory=list)


class OrganizationStat(BaseModel):
    name: str
    documents: int
    passages: int


class ModelStat(BaseModel):
    key: str
    name: str
    hf_id: str
    dim: int
    use: str


class LibraryTotals(BaseModel):
    sources: int
    passages: int
    collections: int


class LibraryResponse(BaseModel):
    totals: LibraryTotals
    collections: list[CollectionStat] = Field(default_factory=list)
    organizations: list[OrganizationStat] = Field(default_factory=list)
    models: list[ModelStat] = Field(default_factory=list)
