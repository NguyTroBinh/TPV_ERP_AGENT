"""
ingestion/worker.py
────────────────────
FastAPI app riêng cho ingestion service.

Chạy trong container riêng (compose.prod.yml → port 8002).
Nginx route /api/upload → container này.

- OCR model (2GB+) + Embedding model (1024 dim) cần VRAM riêng
- App container (agent) không cần load model nặng này
- Có thể scale ingestion service trên máy GPU riêng
- Upload processing lâu (30s-2min) → không block chat requests

Dev mode: api/routes/upload.py gọi trực tiếp ingestion pipeline
          (chạy cùng process, đơn giản hơn)
Prod mode: nginx proxy /api/upload → worker này
"""

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models khi worker start."""
    logger.info("ingestion_worker.starting")

    # Pre-warm embedding model (load vào VRAM 1 lần)
    try:
        from services.embedder import embedding_client
        _ = embedding_client.get_embedding("warm up")
        logger.info("ingestion_worker.embedder_ready")
    except Exception as e:
        logger.error("ingestion_worker.embedder_failed", error=str(e))

    logger.info("ingestion_worker.ready")
    yield
    logger.info("ingestion_worker.shutdown")


app = FastAPI(
    title="ERP Chatbot — Ingestion Worker",
    docs_url="/docs",
    lifespan=lifespan,
)


# ── Mount upload routes ──────────────────────────────
# Reuse cùng route logic với app chính
from api.routes.upload import router as upload_router
app.include_router(upload_router)


# ── Health check riêng cho ingestion ─────────────────
@app.get("/health")
async def health():
    """Health check cho ingestion worker."""
    checks = {}

    # Check Qdrant (cần để upsert vectors)
    try:
        from memory.vectorstore import vector_store_service
        await vector_store_service.health_check()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    # Check embedding model loaded
    try:
        from services.embedder import embedding_client
        test = embedding_client.get_embedding("test")
        checks["embedder"] = "ok" if test else "error: empty"
    except Exception as e:
        checks["embedder"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "healthy" if all_ok else "degraded",
            "service": "ingestion_worker",
            "checks": checks,
        },
    )