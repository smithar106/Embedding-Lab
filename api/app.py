"""FastAPI application — the production transport layer.

The API is ONLY a transport layer. Routes delegate to the existing agent/tool
code and return results — no agent or retrieval logic is duplicated here.

Lifespan: load the embedding model ONCE (reused across requests), verify
database connectivity, and close the pool on shutdown.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from agent.agent import run_agent
from config import get_settings
from database.connection import check_connection, close_pool
from database.library import get_library_stats
from models.embedding_models import MODELS, embed_query, is_model_loaded
from retrieval.retriever import retrieve_compare
from tools.retrieval_tool import retrieval_search

from api.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    ConfigResponse,
    HealthResponse,
    LatencyMs,
    LibraryResponse,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger("embedding_lab.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    # Load the embedding model ONCE and reuse it for every request.
    try:
        embed_query("warmup", settings.embedding_model)
        info = MODELS[settings.embedding_model]
        logger.info("Embedding model loaded; model=%s dim=%d", info.hf_id, info.dim)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load embedding model: %s", exc)

    db_ok = check_connection()
    logger.info("Database %s", "connected" if db_ok else "UNAVAILABLE")

    yield

    close_pool()


app = FastAPI(title="Embedding-Lab", version="1.0.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)

# CORS is configurable via the CORS_ORIGINS env var (comma-separated).
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins.split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
    response.headers["x-request-id"] = request_id
    logger.info(
        json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        })
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log the full error server-side; return a clean response to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    db_ok = check_connection()
    emb_loaded = is_model_loaded(settings.embedding_model)
    healthy = db_ok and emb_loaded
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "unhealthy",
            "database": "connected" if db_ok else "unavailable",
            "embedding_model": "loaded" if emb_loaded else "not loaded",
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    settings = get_settings()
    trace = run_agent(req.question)

    if trace.get("error"):
        logger.error("ask failed for %s: %s", request.state.request_id, trace["error"])
        raise HTTPException(status_code=502, detail="upstream generation error")

    latency = LatencyMs(
        agent_decision=trace["latency"].get("agent_decision_ms", 0.0),
        retrieval=trace["latency"].get("tool_execution_ms", 0.0),
        generation=trace["latency"].get("final_generation_ms", 0.0),
        total=trace["latency"].get("total_ms", 0.0),
    )
    response = AskResponse(
        answer=trace["final_answer"] or "",
        citations=trace["citations"],
        tool_used=trace["tool_requested"],
        retrieved_chunk_ids=[r["chunk_id"] for r in trace["tool_results"]],
        latency_ms=latency,
        token_usage=trace["usage"],
        request_id=request.state.request_id,
    )
    # Log observable execution facts (not chain-of-thought).
    logger.info(
        json.dumps({
            "request_id": request.state.request_id,
            "endpoint": "/ask",
            "tool_used": response.tool_used,
            "retrieval_ms": latency.retrieval,
            "generation_ms": latency.generation,
            "total_ms": latency.total,
            "status": 200,
        })
    )
    return response


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest, request: Request):
    settings = get_settings()
    top_k = req.top_k if req.top_k is not None else settings.top_k
    try:
        result = retrieval_search(req.query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve failed for %s: %s", request.state.request_id, exc)
        raise HTTPException(status_code=502, detail="retrieval failed")
    return RetrieveResponse(
        query=result["query"],
        embedding_model=result["embedding_model"],
        top_k=result["top_k"],
        results=result["results"],
    )


@app.get("/config", response_model=ConfigResponse)
def config_endpoint():
    settings = get_settings()
    return ConfigResponse(
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )


@app.get("/library", response_model=LibraryResponse)
def library_endpoint():
    return get_library_stats()


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest, request: Request):
    top_k = req.top_k if req.top_k is not None else 1
    try:
        groups = retrieve_compare(req.query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("compare failed for %s: %s", request.state.request_id, exc)
        raise HTTPException(status_code=502, detail="comparison failed")
    return CompareResponse(query=req.query, top_k=top_k, groups=groups)
