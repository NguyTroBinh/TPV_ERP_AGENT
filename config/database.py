"""
config/database.py
──────────────────
Singleton asyncpg connection pool cho ERP database.

Dùng:
    from config.database import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT ...")
"""

import asyncpg
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Trả về pool singleton — tạo lần đầu, dùng lại sau đó."""
    global _pool
    if _pool is None or _pool._closed:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            dsn=settings.erp_db.url_sync,
            min_size=2,
            max_size=settings.erp_db.pool_size,
            command_timeout=settings.erp_db.pool_timeout,
        )
        logger.info(
            "db_pool.created",
            min_size=2,
            max_size=settings.erp_db.pool_size,
        )
    return _pool


async def close_pool() -> None:
    """Đóng pool — gọi khi shutdown app."""
    global _pool
    if _pool and not _pool._closed:
        await _pool.close()
        logger.info("db_pool.closed")
        _pool = None
