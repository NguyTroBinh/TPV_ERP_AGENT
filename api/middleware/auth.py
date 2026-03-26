"""
api/middleware/auth.py
──────────────────────
Authentication middleware cho ERP Agentic Chatbot.

2 mode xác thực:
1. API Key (header X-API-Key) — cho internal services, testing
2. JWT Bearer Token — cho frontend users

Sau khi xác thực → extract tenant_id, user_id vào request.state
để các route phía sau dùng.
"""

import structlog
import time
import hmac
import hashlib
from typing import Optional

from fastapi import Request
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response, JSONResponse
from jose import jwt, JWTError

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Các path KHÔNG cần auth
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/chat",
    "/api/upload",
}

security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware xác thực mỗi request.

    Flow:
        Request vào
          → path public? → bypass
          → có X-API-Key header? → validate API key
          → có Bearer token? → validate JWT
          → không có gì? → 401
          → set request.state.user_id, tenant_id
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # ── Public paths — bypass auth ───────────────
        path = request.url.path
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PATHS if p.startswith("/api")):
            return await call_next(request)

        # Preflight CORS
        if request.method == "OPTIONS":
            return await call_next(request)

        # ── Try API Key ──────────────────────────────
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user_info = self._validate_api_key(api_key)
            if user_info:
                request.state.user_id = user_info["user_id"]
                request.state.tenant_id = user_info["tenant_id"]
                request.state.auth_method = "api_key"
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "API key không hợp lệ"})

        # ── Try Bearer Token (JWT) ───────────────────
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_info = await self._validate_jwt(token)
            if user_info:
                request.state.user_id = user_info["user_id"]
                request.state.tenant_id = user_info["tenant_id"]
                request.state.auth_method = "jwt"
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "Token không hợp lệ hoặc hết hạn"})

        # ── Không có credentials ─────────────────────
        return JSONResponse(
            status_code=401,
            content={"detail": "Thiếu authentication. Gửi header X-API-Key hoặc Authorization: Bearer <token>"},
        )

    def _validate_api_key(self, api_key: str) -> Optional[dict]:
        """
        Validate API key.

        Production: lookup key trong database/Redis.
        Hiện tại: so sánh với APP_SECRET_KEY trong .env
        + parse tenant_id từ key format: "tenant_xxx:keyyyy"
        """
        # Simple mode: check master key
        if hmac.compare_digest(api_key, settings.app.secret_key):
            return {
                "user_id": "system",
                "tenant_id": "default",
            }

        # Tenant key format: "{tenant_id}:{random_key}"
        # Production sẽ lookup trong DB
        if ":" in api_key:
            parts = api_key.split(":", 1)
            tenant_id = parts[0]
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # TODO: lookup key_hash trong database
            # row = await db.execute(
            #     "SELECT tenant_id, user_id FROM api_keys WHERE key_hash = :h AND active = true",
            #     {"h": key_hash}
            # )
            # if row:
            #     return {"user_id": row.user_id, "tenant_id": row.tenant_id}

            # Tạm thời: accept nếu format đúng (DEV ONLY)
            if settings.app.env == "development":
                return {
                    "user_id": "dev_user",
                    "tenant_id": tenant_id,
                }

        return None

    async def _validate_jwt(self, token: str) -> Optional[dict]:
        """
        Validate JWT token dùng python-jose.

        JWT payload expected:
        {
            "sub": "user_id",
            "tenant_id": "xxx",
            "exp": 1234567890
        }
        """
        try:
            payload = jwt.decode(
                token,
                settings.app.secret_key,
                algorithms=[settings.auth.jwt_algorithm],
            )

            exp = payload.get("exp", 0)
            if exp < time.time():
                logger.warning("auth.jwt_expired", user_id=payload.get("sub"))
                return None

            return {
                "user_id": payload.get("sub", "unknown"),
                "tenant_id": payload.get("tenant_id", "default"),
            }

        except JWTError as e:
            logger.warning("auth.jwt_invalid", error=str(e))
            return None
        except Exception as e:
            logger.warning("auth.jwt_error", error=str(e))
            return None


def get_current_user(request: Request) -> dict:
    """
    Helper cho route handlers — lấy user info từ request.state.

    Usage trong route:
        @router.post("/chat")
        async def chat(request: Request):
            user = get_current_user(request)
            tenant_id = user["tenant_id"]
    """
    try:
        role_id = int(request.headers.get("X-Role-ID") or getattr(request.state, "role_id", 0))
    except (ValueError, TypeError):
        role_id = 0
    return {
        "user_id": getattr(request.state, "user_id", "unknown"),
        "tenant_id": getattr(request.state, "tenant_id", "default"),
        "role_id": role_id,
        "auth_method": getattr(request.state, "auth_method", "none"),
    }
