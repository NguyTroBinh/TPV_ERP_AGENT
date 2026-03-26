"""
main.py
────────
FastAPI application entry point.

Wiring order:
1. Middleware (auth → rate_limit → CORS)
2. Routes (health, chat)
3. Startup/shutdown events (DB connections)

Chạy:
  Dev:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  Prod: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import get_settings
from config.logging import setup_logging, get_logger

settings = get_settings()

# Khởi tạo logging từ config TRƯỚC KHI dùng logger
setup_logging(
    level=settings.logging.level,
    fmt=settings.logging.format,
    log_file=settings.logging.file,
)
logger = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LIFESPAN — startup / shutdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo connections khi startup, đóng khi shutdown."""
    logger.info("app.starting", env=settings.app.env)

    # ── Startup ──────────────────────────────────────
    from memory.cache import cache_client
    from memory.session_store import session_store
    from config.database import get_pool, close_pool

    try:
        await cache_client.connect()
        logger.info("app.cache_connected")
    except Exception as e:
        logger.warning("app.cache_not_ready", error=str(e))

    try:
        await session_store.connect()
        logger.info("app.session_store_connected")
    except Exception as e:
        logger.warning("app.session_store_not_ready", error=str(e))

    try:
        await get_pool()
        logger.info("app.db_pool_ready")
    except Exception as e:
        logger.warning("app.db_pool_not_ready", error=str(e))

    logger.info("app.ready", port=settings.app.port)

    yield

    # ── Shutdown ─────────────────────────────────────
    logger.info("app.shutting_down")
    try:
        await close_pool()
        await cache_client.disconnect()
        await session_store.disconnect()
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    docs_url="/docs" if settings.app.env == "development" else None,
    redoc_url="/redoc" if settings.app.env == "development" else None,
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MIDDLEWARE (thứ tự: CORS → Auth → RateLimit)
#  Starlette xử lý middleware LIFO — đăng ký ngược
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3. Rate Limit (chạy sau auth, cần tenant_id)
from api.middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# 2. Auth (chạy sau CORS)
from api.middleware.auth import AuthMiddleware
app.add_middleware(AuthMiddleware)

# 1. CORS (chạy đầu tiên)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app.env == "development" else settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-Session-Id",
    ],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.upload import router as upload_router

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
