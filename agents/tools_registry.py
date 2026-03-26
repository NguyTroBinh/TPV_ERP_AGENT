"""
agents/tools_registry.py
─────────────────────────
Registry trung tâm: khai báo tool schema cho LLM + route tool_call đến handler.

Thiết kế:
- LLM nhìn thấy TOOL_SCHEMA (OpenAI function calling format)
- Agent gọi execute() với tool_name + arguments
- Registry route đến đúng handler module

Thêm tool mới = thêm 1 entry vào TOOLS dict. Không sửa agent.py.
"""

import structlog
from typing import Any

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOL DEFINITIONS
#  Mỗi tool = {schema, handler_module, handler_func}
#  - schema: OpenAI function calling format → LLM đọc
#  - handler: lazy import path → tránh load model khi startup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS: dict[str, dict] = {
    "rag_search": {
        "module": "mcp_tools.tools.rag.search",
        "handler": "handle",
        "schema": {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Tìm kiếm tài liệu nội bộ đã được upload bởi người dùng. "
                    "Dùng khi câu hỏi liên quan đến nội dung file, tài liệu, quy trình, hướng dẫn, hoặc kiến thức nội bộ công ty."
                    "KHÔNG dùng cho dữ liệu số liệu thời gian thực như: đơn hàng, tồn kho, doanh thu, ...)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Câu hỏi tìm kiếm — GIỮ NGUYÊN câu hỏi gốc của user, KHÔNG rút gọn hay viết lại",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "erp_db_query": {
        "module": "mcp_tools.tools.erp_database.query",
        "handler": "handle",
        "schema": {
            "type": "function",
            "function": {
                "name": "erp_db_query",
                "description": (
                    "Truy vấn dữ liệu ERP từ database PostgreSQL. "
                    "Dùng khi cần số liệu thời gian thực: đơn hàng, tồn kho, doanh thu, khách hàng, nhân viên, sản phẩm, ..."
                    "KHÔNG dùng cho nội dung tài liệu — dùng rag_search cho việc đó."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Câu hỏi gốc của user (để log + debug)",
                        },
                        "sql": {
                            "type": "string",
                            "description": (
                                "Câu SQL SELECT để truy vấn. "
                                "Chỉ được SELECT, không INSERT/UPDATE/DELETE. "
                                "Phải có WHERE tenant_id = :tenant_id"
                            ),
                        },
                    },
                    "required": ["question", "sql"],
                },
            },
        },
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Cache handler functions sau lần import đầu tiên
_handler_cache: dict[str, Any] = {}


def get_tool_schemas() -> list[dict]:
    """
    Trả về list tool schemas cho LLM.
    Dùng trong llm_client.invoke(tools=get_tool_schemas())
    """
    return [tool["schema"] for tool in TOOLS.values()]


def get_tool_names() -> list[str]:
    """Danh sách tên tools đã đăng ký."""
    return list(TOOLS.keys())


async def execute(tool_name: str, arguments: dict, context: dict) -> dict:
    """
    Thực thi tool theo tên.

    Args:
        tool_name:   "rag_search" | "erp_db_query" | ...
        arguments:   Dict arguments từ LLM tool_call
        context:     {"tenant_id": ..., "user_id": ..., "session_id": ...}

    Returns:
        Dict kết quả từ tool handler

    Raises:
        ValueError: tool_name không tồn tại
        Exception:  lỗi từ handler (agent sẽ bắt và gửi lại LLM)
    """
    if tool_name not in TOOLS:
        logger.warning("tool_registry.unknown_tool", tool_name=tool_name)
        raise ValueError(f"Tool không tồn tại: {tool_name}")

    tool_config = TOOLS[tool_name]
    log = logger.bind(tool=tool_name)

    # Lazy import handler — chỉ load module lần đầu gọi
    handler_fn = _get_handler(tool_config)

    log.info("tool_registry.executing", args_keys=list(arguments.keys()))

    try:
        result = await handler_fn(arguments=arguments, context=context)
        log.info("tool_registry.success")
        return result

    except Exception as e:
        log.error("tool_registry.error", error=str(e))
        return {
            "error": str(e),
            "tool": tool_name,
            "status": "failed",
        }


def _get_handler(tool_config: dict) -> Any:
    """
    Lazy import + cache handler function.
    Lần đầu: import module → lấy function → cache
    Lần sau: trả từ cache
    """
    cache_key = f"{tool_config['module']}.{tool_config['handler']}"

    if cache_key not in _handler_cache:
        import importlib

        module = importlib.import_module(tool_config["module"])
        handler_fn = getattr(module, tool_config["handler"])
        _handler_cache[cache_key] = handler_fn
        logger.info(
            "tool_registry.handler_loaded",
            module=tool_config["module"],
            handler=tool_config["handler"],
        )

    return _handler_cache[cache_key]