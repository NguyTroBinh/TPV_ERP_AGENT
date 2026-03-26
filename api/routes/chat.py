"""
api/routes/chat.py
───────────────────
Chat endpoints — kết nối user request đến Agent.

2 endpoints:
  POST /api/chat         → response đầy đủ (đợi agent xong mới trả)
  POST /api/chat/stream  → SSE streaming từng token real-time

Request body:
{
    "message": "Tổng doanh thu tháng này?",
    "session_id": "optional-uuid"     ← nếu không gửi → tạo mới
}

Response (non-stream):
{
    "reply": "Doanh thu tháng 3...",
    "session_id": "uuid",
    "tool_calls": [...],
    "latency_ms": 1234.5
}
"""

import json
import structlog
import uuid
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.middleware.auth import get_current_user
from agents.agent import agent, AgentRequest, AgentResponse

logger = structlog.get_logger(__name__)

# SSE headers chung — tắt buffering ở mọi tầng proxy
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
router = APIRouter(prefix="/api/chat", tags=["chat"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REQUEST / RESPONSE MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Tin nhắn từ user",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID. Nếu không gửi → tạo session mới.",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: list[dict] = []
    citations: list[dict] = []
    iterations: int = 0
    latency_ms: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST /api/chat — Full response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """
    Chat endpoint — đợi agent xử lý xong mới trả response.
    Phù hợp cho: API integration, testing, batch processing.
    """
    user = get_current_user(request)

    # Session ID: dùng existing hoặc tạo mới
    session_id = body.session_id or str(uuid.uuid4())

    log = logger.bind(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role_id=user["role_id"],
        session_id=session_id,
    )
    log.info("chat.request", message=body.message[:100])

    try:
        agent_request = AgentRequest(
            user_message=body.message,
            session_id=session_id,
            user_id=user["user_id"],
            tenant_id=user["tenant_id"],
            role_id=user["role_id"],
        )

        result: AgentResponse = await agent.run(agent_request)

        log.info("chat.response", latency_ms=f"{result.latency_ms:.0f}")

        return ChatResponse(
            reply=result.reply,
            session_id=result.session_id,
            tool_calls=result.tool_calls_made,
            citations=result.citations,
            iterations=result.iterations,
            latency_ms=result.latency_ms,
        )

    except Exception as e:
        log.error("chat.error", error=str(e))
        raise HTTPException(status_code=500, detail="Lỗi xử lý tin nhắn. Vui lòng thử lại.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST /api/chat/stream — SSE Streaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    Streaming chat — trả từng token qua Server-Sent Events.
    Phù hợp cho: chat UI real-time.

    SSE format:
        data: {"type": "token", "content": "Doanh"}
        data: {"type": "token", "content": " thu"}
        data: {"type": "tool", "name": "rag_search"}
        data: {"type": "done", "session_id": "uuid"}
        data: {"type": "error", "detail": "..."}
    """
    user = get_current_user(request)
    session_id = body.session_id or str(uuid.uuid4())

    log = logger.bind(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role_id=user["role_id"],
        session_id=session_id,
    )
    log.info("chat.stream_start", message=body.message[:100])

    agent_request = AgentRequest(
        user_message=body.message,
        session_id=session_id,
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role_id=user["role_id"],
    )

    async def event_generator():
        """Async generator cho SSE stream."""
        try:
            async for chunk in agent.run_stream(agent_request):
                # agent đã yield JSON string → wrap SSE trực tiếp, bỏ json.loads + json.dumps
                yield f"data: {chunk}\n\n"
        except Exception as e:
            log.error("chat.stream_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Lỗi xử lý. Vui lòng thử lại.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, "X-Session-Id": session_id},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION-BASED ROUTES (cho frontend)
#  POST /api/chat/sessions            → tạo session
#  POST /api/chat/sessions/{id}/messages → gửi tin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionCreateRequest(BaseModel):
    mode: Optional[str] = None
    title: Optional[str] = None


@router.post("/sessions")
async def create_session(body: SessionCreateRequest = SessionCreateRequest(), request: Request = None):
    """Tạo session mới — lưu vào Redis, trả về chatSessionId cho frontend."""
    from memory.session_store import session_store
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")
    user_id = getattr(request.state, "user_id", "anonymous")

    session_id = await session_store.create_session(
        tenant_id=tenant_id,
        user_id=user_id,
        title=body.title or "New Chat",
        mode=body.mode or "chat",
    )
    return {"chatSessionId": session_id, "session_id": session_id}


@router.get("/sessions")
async def list_sessions(request: Request):
    """Danh sách sessions cho sidebar."""
    from memory.session_store import session_store
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")
    sessions = await session_store.get_sessions_by_tenant(tenant_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request):
    """Lấy lịch sử tin nhắn của session (cho khi user click vào session cũ)."""
    from memory.session_store import session_store
    session = await session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session không tồn tại hoặc đã hết hạn")
    messages = [
        {"role": m.role, "content": m.content, "created_at": m.created_at}
        for m in session.messages
        if m.role in ("user", "assistant")
    ]
    return {"session_id": session_id, "messages": messages}


class MessageRequest(BaseModel):
    # frontend gửi messageText, hỗ trợ thêm message/content cho các client khác
    messageText: Optional[str] = Field(default=None, max_length=5000)
    message: Optional[str] = Field(default=None, max_length=5000)
    content: Optional[str] = Field(default=None, max_length=5000)
    use_rag: bool = True
    stream: bool = False
    uploaded_files: list[str] = []  # filenames đã upload trong session → scope RAG search

    def get_message(self) -> str:
        text = self.messageText or self.message or self.content or ""
        if not text.strip():
            raise ValueError("message không được trống")
        return text


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: MessageRequest, request: Request):
    # "undefined" từ frontend → tạo session mới
    if not session_id or session_id == "undefined":
        session_id = str(uuid.uuid4())

    # Đọc tenant_id, accessed_role từ header nếu auth bypass
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")
    user_id = getattr(request.state, "user_id", "default")

    try:
        role_id = int(request.headers.get("X-Role-ID") or getattr(request.state, "role_id", 0))
    except (ValueError, TypeError):
        role_id = 0

    log = logger.bind(user_id=user_id, tenant_id=tenant_id, role_id=role_id, session_id=session_id)
    log.info("chat.session_message", message=body.get_message()[:100])

    try:
        agent_request = AgentRequest(
            user_message=body.get_message(),
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            role_id=role_id,
            uploaded_files=body.uploaded_files or [],
        )

        if body.stream:
            async def event_generator():
                try:
                    async for chunk in agent.run_stream(agent_request):
                        yield f"data: {chunk}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={**_SSE_HEADERS, "X-Session-Id": session_id},
            )

        result: AgentResponse = await agent.run(agent_request)
        # Trả về format frontend mong đợi
        return {
            "assistantMessage": result.reply,
            "citations": result.citations,
            "session_id": result.session_id,
            "chatSessionId": result.session_id,
            "tool_calls": result.tool_calls_made,
            "latency_ms": result.latency_ms,
        }

    except Exception as e:
        log.error("chat.session_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý tin nhắn: {str(e)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST /api/chat/ingest — Upload + Ingest (frontend)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/ingest")
async def ingest_file(request: Request, file: UploadFile = File(...)):
    """Upload file → ingest vào Qdrant. Dùng bởi frontend upload widget."""
    import shutil, uuid as _uuid
    from pathlib import Path

    tenant_id = request.headers.get("X-Tenant-ID", "default")
    
    accessed_role_str = request.headers.get("X-Accessed-Role")
    if accessed_role_str:
        accessed_role = [int(x.strip()) for x in accessed_role_str.split(",") if x.strip().isdigit()]
    else:
        accessed_role = []

    ext = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".doc", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".json", ".csv"}
    if ext not in allowed:
        raise HTTPException(400, f"Định dạng {ext} không hỗ trợ")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(400, "File quá lớn. Tối đa 100MB")

    request_id = str(_uuid.uuid4())
    temp_dir = Path(f"/tmp/ingestion/{request_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        temp_path.write_bytes(content)

        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        result = await pipeline.ingest_file(
            file_path=str(temp_path),
            tenant_id=tenant_id,
            accessed_role=accessed_role,
            metadata={"source": file.filename},
        )

        chunks = result.get("chunks", 0)
        logger.info("ingest.done", filename=file.filename, chunks=chunks, tenant=tenant_id)
        return {"status": "success", "filename": file.filename, "chunks_processed": chunks}

    except Exception as e:
        logger.error("ingest.error", error=str(e))
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)