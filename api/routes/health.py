"""
api/routes/health.py
─────────────────────
Health check endpoint — Docker healthcheck gọi mỗi 30s.

GET /health → 200 nếu tất cả services OK
            → 503 nếu bất kỳ service nào down

Response:
{
    "status": "healthy",
    "services": {
        "postgres": {"status": "ok", "latency_ms": 2.1},
        "redis":    {"status": "ok", "latency_ms": 0.8},
        "qdrant":   {"status": "ok", "latency_ms": 3.2},
    }
}
"""

import structlog
import time

from fastapi import APIRouter

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check tổng hợp — check tất cả dependencies.
    Trả 200 nếu OK, 503 nếu bất kỳ service nào fail.
    """
    services = {}
    overall_healthy = True

    # ── PostgreSQL ───────────────────────────────────
    services["postgres"] = await _check_postgres()
    if services["postgres"]["status"] != "ok":
        overall_healthy = False

    # ── Redis ────────────────────────────────────────
    services["redis"] = await _check_redis()
    if services["redis"]["status"] != "ok":
        overall_healthy = False

    # ── Qdrant ───────────────────────────────────────
    services["qdrant"] = await _check_qdrant()
    if services["qdrant"]["status"] != "ok":
        overall_healthy = False

    status_code = 200 if overall_healthy else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if overall_healthy else "degraded",
            "services": services,
        },
    )


async def _check_postgres() -> dict:
    """Check PostgreSQL connection."""
    start = time.monotonic()
    try:
        from memory.session_store import session_store
        await session_store.health_check()
        latency = (time.monotonic() - start) * 1000
        return {"status": "ok", "latency_ms": round(latency, 1)}
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        logger.error("health.postgres_fail", error=str(e))
        return {"status": "error", "error": str(e), "latency_ms": round(latency, 1)}


async def _check_redis() -> dict:
    """Check Redis connection."""
    start = time.monotonic()
    try:
        from memory.cache import cache_client
        result = await cache_client.ping()
        latency = (time.monotonic() - start) * 1000
        if result:
            return {"status": "ok", "latency_ms": round(latency, 1)}
        return {"status": "error", "error": "ping failed"}
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        logger.error("health.redis_fail", error=str(e))
        return {"status": "error", "error": str(e), "latency_ms": round(latency, 1)}


async def _check_qdrant() -> dict:
    """Check Qdrant connection."""
    start = time.monotonic()
    try:
        from memory.vectorstore import vector_store_service
        await vector_store_service.health_check()
        latency = (time.monotonic() - start) * 1000
        return {"status": "ok", "latency_ms": round(latency, 1)}
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        logger.error("health.qdrant_fail", error=str(e))
        return {"status": "error", "error": str(e), "latency_ms": round(latency, 1)}