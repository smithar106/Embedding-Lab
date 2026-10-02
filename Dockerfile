# Production image for Embedding-Lab.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Where the embedding model is downloaded/cached (configurable at runtime).
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Minimal system deps (curl is handy for health/debug; nothing else is needed —
# psycopg[binary] and torch ship their own wheels).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Bind to $PORT on Railway, falling back to 8000 locally.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
