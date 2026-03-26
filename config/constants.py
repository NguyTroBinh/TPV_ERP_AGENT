from enum import StrEnum


# ── ReAct Loop ────────────────────────────────────────────────
class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"


# ── MCP Tool names ────────────────────────────────────────────
class ToolName(StrEnum):
    ERP_QUERY = "erp_query"          # query ERP database
    ERP_SCHEMA = "erp_schema"        # lấy schema bảng
    RAG_SEARCH = "rag_search"        # tìm kiếm tài liệu
    RAG_LIST = "rag_list"            # liệt kê tài liệu có sẵn


# ── Message roles ─────────────────────────────────────────────
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ── HTTP headers ──────────────────────────────────────────────
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"

# ── Session ───────────────────────────────────────────────────
SESSION_KEY_PREFIX = "session:"
CACHE_KEY_PREFIX = "cache:"

# ── Pagination ────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# ── RAG ───────────────────────────────────────────────────────
RAG_TOP_K = 5                        # số chunk trả về mỗi lần search
RAG_SCORE_THRESHOLD = 0.7            # min similarity score

# ── ERP ───────────────────────────────────────────────────────
ERP_QUERY_ROW_LIMIT = 500            # giới hạn rows mỗi query
ERP_ALLOWED_SCHEMAS = ["public", "inventory", "sales", "purchase"]