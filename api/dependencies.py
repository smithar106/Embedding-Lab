"""Shared API dependencies: configuration + a simple rate-limiter structure.

The rate limiter is intentionally basic and DISABLED by default (no env
configured). It exists so production can enable it with RATE_LIMIT_MAX /
RATE_LIMIT_WINDOW without code changes — the public endpoint should not be
treated as a high-volume unrestricted service without rate limiting/auth.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from config import get_settings


def get_config():
    return get_settings()


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, max_requests: int, window_seconds: int) -> bool:
        if max_requests <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > window_seconds:
                dq.popleft()
            if len(dq) >= max_requests:
                return False
            dq.append(now)
            return True


_limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    settings = get_settings()
    max_req = int(os.environ.get("RATE_LIMIT_MAX", "0"))
    window = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
    if max_req <= 0:
        return
    if not _limiter.allow(client_ip(request), max_req, window):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
