from typing import Any
from mcp.server.fastmcp import FastMCP
from config import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Khởi tạo MCP Server ───────────────────────────────────────
mcp = FastMCP(
    name="erp-chatbot-mcp",
    instructions=(
        "MCP server cho ERP Agentic Chatbot. "
        "Cung cấp tools truy vấn ERP database và tìm kiếm tài liệu RAG."
    ),
)


# ── Tool: ERP Database ────────────────────────────────────────
@mcp.tool()
async def erp_db_query(question: str, sql: str, tenant_id: str) -> dict[str, Any]:
    """
    Truy vấn dữ liệu ERP từ database PostgreSQL (READ-ONLY).

    Args:
        question:  Câu hỏi gốc của user (để log + debug)
        sql:       Câu SELECT — chỉ SELECT, phải có WHERE tenant_id = :tenant_id
    Context:
        tenant_id: ID tenant để kiểm soát quyền truy cập

    Returns:
        {"rows": [...], "columns": [...], "row_count": int}
    """
    from mcp_tools.tools.erp_database.query import handle
    return await handle(
        arguments={"question": question, "sql": sql},
        context={"tenant_id": tenant_id},
    )


# ── Tool: RAG ─────────────────────────────────────────────────
@mcp.tool()
async def rag_search_tool(query: str, tenant_id: str, uploaded_files: list[str] = None, role_id: int = None) -> dict[str, Any]:
    """
    Tìm kiếm tài liệu nội bộ liên quan đến câu hỏi (hybrid search).
    Dùng khi cần tra cứu quy định, chính sách, hướng dẫn nghiệp vụ.

    Args:
        query:     Câu hỏi hoặc từ khóa cần tìm
    Context:
        tenant_id: ID tenant để lọc đúng tài liệu
        role_id:   Quyền truy cập vào tài liệu của User
        uploaded_files: Các file đã được User upload

    Returns:
        {"results": [{"content": "...", "score": float, "metadata": {...}}]}
    """
    from mcp_tools.tools.rag.search import handle
    return await handle(
        arguments={"query": query},
        context={"tenant_id": tenant_id, "role_id": role_id, "uploaded_files": uploaded_files}
    )


@mcp.tool()
async def rag_list_documents(tenant_id: str) -> dict[str, Any]:
    """
    Liệt kê danh sách tài liệu đã được index cho tenant.
    Dùng khi cần biết hệ thống đang có những tài liệu gì.

    Args:
        tenant_id: ID tenant

    Returns:
        {"documents": [{"filename": "...", "chunk_count": int}]}
    """
    from mcp_tools.tools.rag.search import list_documents
    return await list_documents(tenant_id=tenant_id)


# ── Entry point (chạy standalone) ────────────────────────────
if __name__ == "__main__":
    logger.info(
        "mcp_server_starting",
        host=settings.mcp.host,
        port=settings.mcp.port,
    )
    mcp.run(transport="sse")
