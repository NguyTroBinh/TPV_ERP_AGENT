"""
api/routes/upload.py
─────────────────────
Upload endpoints — user tải file lên → ingest vào Qdrant.

Endpoints:
  POST   /api/upload              upload + ingest 1 file
  GET    /api/upload/documents    danh sách file đã index
  DELETE /api/upload/{filename}   xóa file khỏi vector DB

Luồng:
  User upload file
    → validate (extension, size)
    → save temp file
    → gọi ingestion pipeline (convert → chunk → embed → Qdrant)
    → track trong bảng documents
    → xóa temp file
"""

import structlog
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from config import get_settings
from config.database import get_pool

logger = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".json", ".csv"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESPONSE MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int = 0
    message: str = ""


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    status: str
    uploaded_at: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST /api/upload — Upload + Ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("", response_model=UploadResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload 1 file → trả response ngay → chạy ingestion pipeline ở background.
    UI theo dõi trạng thái qua GET /api/upload/documents.
    """
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")
    user_id = getattr(request.state, "user_id", "Binh123")
    role_header = request.headers.get("X-Accessed-Role")
    try:
        accessed_role = [int(r.strip()) for r in role_header.split(',') if r.strip().isdigit()] if role_header else []
    except (ValueError, TypeError):
        accessed_role = []

    log = logger.bind(
        tenant_id=tenant_id,
        user_id=user_id,
        accessed_role=accessed_role,
        filename=file.filename,
    )

    # ── Validate ─────────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "Thiếu tên file")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Định dạng {ext} không hỗ trợ. Cho phép: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File quá lớn. Tối đa {MAX_FILE_SIZE // (1024*1024)}MB")
    await file.seek(0)

    # ── Save temp file ───────────────────────────────
    request_id = str(uuid.uuid4())
    temp_dir = Path(f"/tmp/ingestion/{request_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log.info("upload.file_saved", size=len(content))
    except Exception as e:
        log.error("upload.save_error", error=str(e))
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(500, f"Lỗi lưu file: {str(e)}")

    # ── Track document in DB (status = processing) ───
    doc_id = await _track_document(
        tenant_id=tenant_id,
        filename=file.filename,
        file_size=len(content),
        uploaded_by=user_id,
    )

    # ── Chạy pipeline ở background → trả response ngay ──
    background_tasks.add_task(
        _run_pipeline_background,
        temp_dir=temp_dir,
        temp_path=temp_path,
        tenant_id=tenant_id,
        user_id=user_id,
        accessed_role=accessed_role,
        filename=file.filename,
        doc_id=doc_id,
    )

    return UploadResponse(
        status="processing",
        filename=file.filename,
        message="File đang được xử lý. Theo dõi trạng thái tại danh sách tài liệu.",
    )


PIPELINE_TIMEOUT = 600  # 10 phút — tránh document stuck "processing" mãi


async def _run_pipeline_background(
    temp_dir: Path,
    temp_path: Path,
    tenant_id: str,
    user_id: str,
    accessed_role: List[int],
    filename: str,
    doc_id: str,
) -> None:
    """Chạy ingestion pipeline trong background task."""
    import asyncio

    log = logger.bind(tenant_id=tenant_id, filename=filename, accessed_role=accessed_role)
    try:
        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

        result = await asyncio.wait_for(
            pipeline.ingest_file(
                file_path=str(temp_path),
                tenant_id=tenant_id,
                accessed_role=accessed_role,
                metadata={"uploaded_by": user_id},
            ),
            timeout=PIPELINE_TIMEOUT,
        )

        if result["status"] == "success":
            await _update_document_status(
                doc_id=doc_id,
                status="indexed",
                chunk_count=result["chunks"],
            )
            log.info("upload.done", chunks=result["chunks"])
        else:
            await _update_document_status(
                doc_id=doc_id,
                status="failed",
                error=result.get("reason", "unknown"),
            )
            log.warning("upload.skipped", reason=result.get("reason"))

    except asyncio.TimeoutError:
        log.error("upload.pipeline_timeout", timeout=PIPELINE_TIMEOUT)
        if doc_id:
            await _update_document_status(doc_id=doc_id, status="failed", error=f"Timeout sau {PIPELINE_TIMEOUT}s")

    except Exception as e:
        log.error("upload.pipeline_error", error=str(e))
        if doc_id:
            await _update_document_status(doc_id=doc_id, status="failed", error=str(e))

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GET /api/upload/status/{filename} — Check processing status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/status/{filename}")
async def check_upload_status(filename: str, request: Request):
    """Check trạng thái xử lý file — frontend poll endpoint này sau upload."""
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, chunk_count, error_message
                FROM documents
                WHERE tenant_id = $1 AND filename = $2 AND status != 'deleted'
                ORDER BY created_at DESC LIMIT 1
                """,
                tenant_id, filename,
            )

        if not row:
            return {"status": "not_found", "filename": filename}

        return {
            "status": row["status"],
            "filename": filename,
            "chunks": row["chunk_count"] or 0,
            "error": row["error_message"] or "",
        }
    except Exception as e:
        logger.error("upload.status_error", error=str(e))
        return {"status": "error", "filename": filename}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GET /api/upload/documents — List documents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/documents")
async def list_documents(request: Request):
    """Danh sách file đã upload + trạng thái index."""
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT filename, chunk_count, status, created_at
                FROM documents
                WHERE tenant_id = $1 AND status != 'deleted'
                ORDER BY created_at DESC
                LIMIT 100
                """,
                tenant_id,
            )

        return {
            "documents": [
                {
                    "filename": r["filename"],
                    "chunk_count": r["chunk_count"],
                    "status": r["status"],
                    "uploaded_at": r["created_at"].isoformat() if r["created_at"] else "",
                }
                for r in rows
            ],
            "total": len(rows),
        }

    except Exception as e:
        logger.error("upload.list_error", error=str(e))
        raise HTTPException(500, "Lỗi truy vấn danh sách tài liệu")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GET /api/upload/docs — List documents from Qdrant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/docs")
async def list_docs_qdrant(request: Request):
    """Danh sách tài liệu đã index trong Qdrant — dùng cho sidebar UI."""
    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")

    try:
        from mcp_tools.tools.rag.search import list_documents
        result = await list_documents(tenant_id=tenant_id)
        return result
    except Exception as e:
        logger.error("upload.list_docs_error", error=str(e))
        return {"documents": [], "total": 0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DELETE /api/upload/{filename} — Remove document
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.delete("/{filename}")
async def delete_document(filename: str, request: Request):
    """Xóa file khỏi Qdrant + đánh dấu deleted trong DB."""

    tenant_id = request.headers.get("X-Tenant-ID") or getattr(request.state, "tenant_id", "default")

    log = logger.bind(tenant_id=tenant_id, filename=filename)
    log.info("upload.delete_start")

    try:
        # Xóa vectors từ Qdrant (chỉ xóa đúng tenant + file + role)
        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        result = await pipeline.delete_file(tenant_id=tenant_id, filename=filename)

        # Update DB status
        await _update_document_status_by_name(
            tenant_id=tenant_id,
            filename=filename,
            status="deleted",
        )

        log.info("upload.deleted", result=result)
        return {
            "status": "deleted",
            "filename": filename,
            "deleted_chunks": result.get("deleted_count", 0),
        }

    except Exception as e:
        log.error("upload.delete_error", error=str(e))
        raise HTTPException(500, f"Lỗi xóa file: {str(e)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DB HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _track_document(
    tenant_id: str, filename: str, file_size: int, uploaded_by: str
) -> str:
    """Insert document record, return doc_id."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (tenant_id, filename, file_size, uploaded_by, status)
                VALUES ($1, $2, $3, $4, 'processing')
                ON CONFLICT (tenant_id, filename) WHERE status != 'deleted'
                DO UPDATE SET status = 'processing', file_size = $3, created_at = NOW()
                RETURNING id
                """,
                tenant_id, filename, file_size, uploaded_by,
            )
            return str(row["id"]) if row else ""
    except Exception as e:
        logger.error("upload.track_error", error=str(e))
        return ""


async def _update_document_status(
    doc_id: str, status: str, chunk_count: int = 0, error: str = ""
) -> None:
    """Update document status after ingestion."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET status = $1::varchar,
                    chunk_count = $2,
                    error_message = $3,
                    indexed_at = CASE WHEN $1::varchar = 'indexed' THEN NOW() ELSE indexed_at END
                WHERE id = $4::uuid
                """,
                status, chunk_count, error, doc_id,
            )
    except Exception as e:
        logger.error("upload.update_status_error", error=str(e))


async def _update_document_status_by_name(
    tenant_id: str, filename: str, status: str
) -> None:
    """Update document status by filename."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents SET status = $1
                WHERE tenant_id = $2 AND filename = $3 AND status != 'deleted'
                """,
                status, tenant_id, filename,
            )
    except Exception as e:
        logger.error("upload.update_by_name_error", error=str(e))