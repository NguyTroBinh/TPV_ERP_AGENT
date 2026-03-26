import json
import hashlib
from typing import Any, Optional
from redis.asyncio import Redis, ConnectionPool
from config import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()


class CacheService:
    """
    Redis cache dùng cho:
    - Kết quả search RAG (tránh embed + query lại cùng 1 câu)
    - Kết quả query ERP DB (tránh hit DB liên tục với cùng câu hỏi)
    - Tool schema list từ MCP server
    """

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._pool = ConnectionPool.from_url(
            settings.redis.url,
            max_connections=20,
            decode_responses=True,
        )
        self._client = Redis(connection_pool=self._pool)
        # kiểm tra kết nối ngay
        await self._client.ping()
        logger.info("cache_connected", url=settings.redis.url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
        logger.info("cache_disconnected")

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("CacheService chưa được connect(). Gọi await cache.connect() trước.")
        return self._client

    # ── Internal helpers ──────────────────────────────────────

    def _make_key(self, namespace: str, raw: str) -> str:
        """
        Tạo cache key ngắn gọn, an toàn.
        namespace:rag → rag:<md5(raw)>
        """
        digest = hashlib.md5(raw.encode()).hexdigest()
        return f"{settings.redis.url and 'cache'}:{namespace}:{digest}"

    # ── Generic get/set/delete ────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        raw = await self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or settings.redis.cache_ttl
        serialized = json.dumps(value, ensure_ascii=False)
        await self.client.setex(key, ttl, serialized)

    async def delete(self, key: str) -> None:
        await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self.client.exists(key))

    # ── RAG search cache ──────────────────────────────────────

    def _rag_key(self, tenant_id: str, query: str, k: int) -> str:
        return self._make_key("rag", f"{tenant_id}:{query}:{k}")

    async def get_rag_result(
        self, tenant_id: str, query: str, k: int
    ) -> Optional[list]:
        key = self._rag_key(tenant_id, query, k)
        result = await self.get(key)
        if result is not None:
            logger.debug("cache_hit", namespace="rag", tenant=tenant_id)
        return result

    async def set_rag_result(
        self, tenant_id: str, query: str, k: int, results: list
    ) -> None:
        key = self._rag_key(tenant_id, query, k)
        await self.set(key, results, ttl=settings.redis.cache_ttl)

    # ── ERP query cache ───────────────────────────────────────

    def _erp_key(self, tenant_id: str, sql: str) -> str:
        return self._make_key("erp", f"{tenant_id}:{sql}")

    async def get_erp_result(
        self, tenant_id: str, sql: str
    ) -> Optional[list]:
        key = self._erp_key(tenant_id, sql)
        result = await self.get(key)
        if result is not None:
            logger.debug("cache_hit", namespace="erp", tenant=tenant_id)
        return result

    async def set_erp_result(
        self, tenant_id: str, sql: str, rows: list, ttl: int = 60
    ) -> None:
        """ERP data thay đổi thường xuyên → ttl ngắn hơn RAG (default 60s)."""
        key = self._erp_key(tenant_id, sql)
        await self.set(key, rows, ttl=ttl)

    # ── MCP tool schema cache ─────────────────────────────────

    MCP_SCHEMA_KEY = "cache:mcp:tool_schemas"

    async def get_tool_schemas(self) -> Optional[list]:
        return await self.get(self.MCP_SCHEMA_KEY)

    async def set_tool_schemas(self, schemas: list, ttl: int = 300) -> None:
        """Schema ít thay đổi → cache 5 phút."""
        await self.set(self.MCP_SCHEMA_KEY, schemas, ttl=ttl)

    async def invalidate_tool_schemas(self) -> None:
        await self.delete(self.MCP_SCHEMA_KEY)

    # ── Bulk invalidate theo tenant ───────────────────────────

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """
        Xóa toàn bộ cache RAG + ERP của 1 tenant.
        Gọi sau khi tenant upload tài liệu mới.
        """
        pattern = f"cache:*:{hashlib.md5(tenant_id.encode()).hexdigest()[:8]}*"
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)
        logger.info("cache_invalidated", tenant=tenant_id, keys_deleted=len(keys))
        return len(keys)


# Singleton
cache_client = CacheService()