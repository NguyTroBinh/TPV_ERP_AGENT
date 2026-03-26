"""
api/middleware/rate_limit.py
─────────────────────────────
Rate limiting bằng Redis — sliding window per tenant.

Thuật toán: Sliding Window Counter
- Key: rate_limit:{tenant_id}:{window_minute}
- TTL: 120s (tự xóa sau 2 phút)
- Mỗi request tăng counter
- Vượt limit → 429 Too Many Requests

Tại sao không dùng thư viện (slowapi)?
- Cần rate limit per tenant, không per IP
- Logic đơn giản (~40 dòng), không cần dependency thêm
- Dùng chung Redis connection pool đã có
"""

import structlog
import time

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Paths không rate limit
EXEMPT_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limit per tenant dùng Redis sliding window.

    Headers trả về:
    - X-RateLimit-Limit: số request tối đa / phút
    - X-RateLimit-Remaining: số request còn lại
    - X-RateLimit-Reset: timestamp reset window
    - Retry-After: seconds (khi bị 429)
    """

    def __init__(self, app, rpm: int = None):
        super().__init__(app)
        self.rpm = rpm or settings.rate_limit.per_minute
        self._redis = None

    async def _get_redis(self):
        """Lazy init Redis connection."""
        if self._redis is None:
            from memory.cache import cache_client
            self._redis = cache_client
        return self._redis

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # ── Exempt paths ─────────────────────────────
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        # ── Get tenant_id (set by auth middleware) ───
        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            # Auth middleware chưa chạy hoặc bypass → dùng IP
            tenant_id = request.client.host if request.client else "unknown"

        # ── Check rate limit ─────────────────────────
        try:
            current_minute = int(time.time()) // 60
            key = f"rate_limit:{tenant_id}:{current_minute}"
            window_reset = (current_minute + 1) * 60

            redis = await self._get_redis()
            current_count = await redis.client.incr(key)

            # Set TTL chỉ lần đầu (khi counter = 1)
            if current_count == 1:
                await redis.client.expire(key, 120)

            remaining = max(0, self.rpm - current_count)

            # ── Over limit → 429 ────────────────────
            if current_count > self.rpm:
                retry_after = window_reset - int(time.time())
                logger.warning(
                    "rate_limit.exceeded",
                    tenant_id=tenant_id,
                    count=current_count,
                    limit=self.rpm,
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Tối đa {self.rpm} requests/phút.",
                    headers={
                        "X-RateLimit-Limit": str(self.rpm),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(window_reset),
                        "Retry-After": str(max(retry_after, 1)),
                    },
                )

            # ── Under limit → proceed ───────────────
            response = await call_next(request)

            # Inject rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.rpm)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(window_reset)

            return response

        except HTTPException:
            raise
        except Exception as e:
            # Redis down → cho request qua (fail open)
            # Tốt hơn là block user vì lỗi infra
            logger.error("rate_limit.redis_error", error=str(e))
            return await call_next(request)