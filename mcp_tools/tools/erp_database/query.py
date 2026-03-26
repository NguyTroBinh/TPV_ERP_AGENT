"""
mcp/tools/erp_database/query.py
─────────────────────────────────
Tool ERP Database Query — thực thi SQL READ-ONLY trên PostgreSQL.

Bảo mật:
- Chỉ cho phép SELECT
- LUÔN kiểm tra WHERE tenant_id để cô lập dữ liệu
- Giới hạn tối đa 500 rows / query
"""

import re
import structlog
import uuid
import decimal
from datetime import date, datetime, time
from typing import Any

from config.constants import ERP_QUERY_ROW_LIMIT
from config.database import get_pool

logger = structlog.get_logger(__name__)

# Chỉ cho phép DML này
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _safe_value(v: Any) -> Any:
    """Convert DB types (UUID, Decimal, datetime...) → JSON-serializable."""
    if v is None:
        return None
    if isinstance(v, (uuid.UUID,)):
        return str(v)
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.hex()
    return v


def _validate_sql(sql: str) -> str | None:
    """
    Kiểm tra SQL an toàn.
    Trả về thông báo lỗi nếu không hợp lệ, None nếu OK.
    """
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return "Chỉ cho phép câu lệnh SELECT."

    if _FORBIDDEN_KEYWORDS.search(sql):
        return "SQL chứa từ khóa không được phép (INSERT/UPDATE/DELETE/DROP...)."

    if "TENANT_ID" not in sql_upper:
        return "SQL phải có điều kiện WHERE tenant_id để cô lập dữ liệu."

    return None


async def _run_query(sql: str, tenant_id: str) -> dict[str, Any]:
    """
    Thực thi SQL trên ERP database.
    Trả về dict chuẩn cho agent.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Bind tenant_id như tham số để tránh SQL injection
            # SQL từ LLM dùng :tenant_id → thay bằng $1 cho asyncpg
            sql_pg = sql.replace(":tenant_id", "$1")

            rows = await conn.fetch(sql_pg, tenant_id)

            if not rows:
                return {
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "status": "empty",
                }

            columns = list(rows[0].keys())
            data = [[_safe_value(v) for v in row.values()] for row in rows[:ERP_QUERY_ROW_LIMIT]]

            return {
                "rows": data,
                "columns": columns,
                "row_count": len(rows),
                "status": "ok",
            }

    except Exception as e:
        logger.error("erp_query.error", error=str(e))
        return {"error": str(e), "status": "failed"}


async def handle(arguments: dict, context: dict) -> dict:
    """
    Entry point cho tools_registry.
    tools_registry.py route tool_name="erp_db_query" → đây.

    Args:
        arguments: {"question": "...", "sql": "SELECT ..."}
        context:   {"tenant_id": "..."}
    """
    question = arguments.get("question", "")
    sql = arguments.get("sql", "").strip()
    tenant_id = context.get("tenant_id", "")

    log = logger.bind(tenant_id=tenant_id)

    if not sql:
        return {"error": "SQL không được để trống.", "status": "failed"}

    if not tenant_id:
        return {"error": "Thiếu tenant_id trong context.", "status": "failed"}

    # Validate SQL
    err = _validate_sql(sql)
    if err:
        log.warning("erp_query.invalid_sql", reason=err, sql=sql, question=question)
        return {"error": err, "status": "failed"}

    log.info("erp_query.executing", sql=sql, question=question)

    result = await _run_query(sql=sql, tenant_id=tenant_id)

    if result.get("status") == "ok":
        log.info(
            "erp_query.success",
            row_count=result.get("row_count", 0),
            columns=result.get("columns", []),
            sql=sql,
        )
    elif result.get("status") == "empty":
        log.info("erp_query.empty", sql=sql)
    else:
        log.warning("erp_query.failed", error=result.get("error"), sql=sql)

    return result
