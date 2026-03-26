"""
mcp/tools/rag/search.py
────────────────────────
Tool RAG Search cho Agent — kết nối toàn bộ pipeline:
  vectorstore (Qdrant hybrid) → reranker → cache

Agent gọi duy nhất: rag_search(query, tenant_id, top_k)
"""

import asyncio
import structlog
import hashlib
from typing import Any

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CACHE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAG_CACHE_TTL = settings.redis.cache_ttl  # default 300s


def _cache_key(tenant_id: str, query: str, top_k: int) -> str:
    """Tạo cache key deterministic từ (tenant, query, k)."""

    raw = f"rag:{tenant_id}:{query.strip().lower()}:{top_k}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"rag_cache:{tenant_id}:{h}"


async def _get_cached(cache, key: str) -> list[dict] | None:
    """Lấy kết quả từ Redis cache. Trả None nếu miss."""

    try:
        result = await cache.get(key)   # CacheService.get() tự json.loads
        return result                   # None nếu miss, list nếu hit
    except Exception as e:
        logger.warning("rag_cache.get_error", error=str(e))
    return None


async def _set_cached(cache, key: str, results: list[dict]) -> None:
    """Lưu kết quả vào Redis cache."""

    try:
        await cache.set(key, results, ttl=RAG_CACHE_TTL)  # CacheService.set() tự json.dumps
    except Exception as e:
        logger.warning("rag_cache.set_error", error=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONTENT EXTRACTION — xử lý cả Qdrant ScoredPoint lẫn dict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_content(doc: Any) -> str:
    """Lấy text content từ ScoredPoint (Qdrant) hoặc dict."""

    if isinstance(doc, dict):
        return doc.get("content", doc.get("text", ""))
    if hasattr(doc, "payload") and doc.payload:
        return doc.payload.get("content", "")
    return getattr(doc, "page_content", "")


def _extract_metadata(doc: Any) -> dict:
    """Lấy metadata từ ScoredPoint hoặc dict."""

    if isinstance(doc, dict):
        return doc.get("metadata", {})
    if hasattr(doc, "payload") and doc.payload:
        return {k: v for k, v in doc.payload.items() if k != "content"}
    return {}


def _extract_score(doc: Any) -> float:
    """Lấy rerank_score (hoặc Qdrant score gốc)."""

    if isinstance(doc, dict):
        return doc.get("rerank_score", doc.get("score", 0.0))
    return getattr(doc, "score", 0.0)


def _format_result(doc: Any) -> dict:
    """Chuẩn hóa kết quả thành dict sạch cho agent."""

    return {
        "content": _extract_content(doc),
        "score": round(_extract_score(doc), 4),
        "metadata": _extract_metadata(doc),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN SEARCH FUNCTION — entry point duy nhất
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def rag_search(query: str, tenant_id: str, filenames: list[str] = None, role_id: int = None) -> dict:
    """
    Pipeline tìm kiếm RAG hoàn chỉnh:

        query
          │
          ├─ 1. Cache hit? → trả về ngay
          │
          ├─ 2. Qdrant hybrid search (dense + BM25 sparse) — chạy qua asyncio.to_thread
          │     VectorStoreService tự embed query bên trong
          │
          ├─ 3. Reranker (Vietnamese_Reranker)
          │     Cross-encoder scoring → sort → cắt top_k
          │
          └─ 4. Cache result + trả về agent
    """
    # Lazy imports — tránh load model khi startup
    from services.reranker import reranker_service
    from memory.vectorstore import vector_store_service
    from memory.cache import cache_client

    RETRIEVE_K = 20     # Lấy 20 candidates từ Qdrant
    RERANK_K = 5        # Reranker chọn top 5

    log = logger.bind(tenant_id=tenant_id, query=query, scoped_files=len(filenames) if filenames else 0)

    # ── 1. Cache check ──────────────────────────────────
    scope_suffix = "|".join(sorted(filenames)) if filenames else ""
    role_suffix = f"|r:{role_id}" if role_id else ""
    cache_key = _cache_key(tenant_id, query + scope_suffix + role_suffix, RERANK_K)
    cached = await _get_cached(cache_client, cache_key)
    if cached is not None:
        log.info("rag_search.cache_hit", results=len(cached))
        return {
            "results": cached,
            "total": len(cached),
            "cached": True,
            "query": query,
        }

    # ── 2. Qdrant hybrid search (dense + BM25 sparse) ──
    candidates = await asyncio.to_thread(
        vector_store_service.search_hybrid,
        query=query,
        tenant_id=tenant_id,   # lọc theo tenant
        filenames=filenames,   # scope search vào files đã upload (focus file)
        role_id=role_id,       # lọc theo quyền xem tài liệu
        k=RETRIEVE_K,          # số candidates trước rerank
    )

    if not candidates:
        log.info("rag_search.no_candidates")
        return {"results": [], "total": 0, "cached": False, "query": query}

    log.info("rag_search.candidates", count=len(candidates))

    # ── 3. Rerank → top 5 ────────────────────────────────
    reranked = await asyncio.to_thread(
        reranker_service.rerank,
        query,
        candidates,
        RERANK_K,
    )

    # ── 4. Format + Cache ───────────────────────────────
    # Trả về ĐÚNG top_k chunks cao nhất sau rerank, KHÔNG lọc theo score
    results = [_format_result(doc) for doc in reranked]

    all_scores = [r["score"] for r in results]
    log.info("rag_search.results", scores=all_scores, count=len(results))

    await _set_cached(cache_client, cache_key, results)

    log.info("rag_search.done", results=len(results))
    return {
        "results": results,
        "total": len(results),
        "cached": False,
        "query": query,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LIST DOCUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def list_documents(tenant_id: str) -> dict:
    """
    Liệt kê danh sách tài liệu đã được index cho tenant.
    Scroll qua toàn bộ Qdrant (phân trang), gom nhóm theo filename.
    """
    from memory.vectorstore import vector_store_service
    from qdrant_client import models

    try:
        client = vector_store_service.client
        collection = vector_store_service.collection_name
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
        )

        # Scroll toàn bộ — phân trang để không giới hạn 1000
        file_counts: dict[str, int] = {}
        next_offset = None

        while True:
            results, next_offset = await asyncio.to_thread(
                client.scroll,
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=next_offset,
                with_payload=["filename"],
                with_vectors=False,
            )

            for point in results:
                if point.payload:
                    fname = point.payload.get("filename", "unknown")
                    file_counts[fname] = file_counts.get(fname, 0) + 1

            if next_offset is None or not results:
                break

        documents = [
            {"filename": fname, "chunk_count": count}
            for fname, count in sorted(file_counts.items())
        ]

        return {"documents": documents, "total": len(documents)}

    except Exception as e:
        logger.error("list_documents.error", error=str(e), tenant=tenant_id)
        return {"documents": [], "total": 0, "error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOL HANDLER — tools_registry gọi hàm này
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def handle(arguments: dict, context: dict) -> dict:
    """
    Entry point cho tools_registry.
    tools_registry.py route tool_name="rag_search" → đây.

    Args:
        arguments:  {"query": "..."}  — từ LLM tool_call
        context:    {"tenant_id": "...", "role_id": "...", "uploaded_files": "..."}  — từ session
    """
    query = arguments.get("query", "").strip()
    tenant_id = context.get("tenant_id")
    role_id = context.get("role_id") or None
    uploaded_files = context.get("uploaded_files") or None

    if not query:
        return {
            "results": [],
            "total": 0,
            "cached": False,
            "query": "",
            "error": "Query không được để trống",
        }

    return await rag_search(
        query=query,
        tenant_id=tenant_id,
        filenames=uploaded_files,
        role_id=role_id
    )
