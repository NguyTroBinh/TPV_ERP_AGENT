import asyncio
import structlog
from pathlib import Path
from typing import Optional, List

from ingestion.converter import document_converter
from ingestion.splitter import ingestion_service
from memory.vectorstore import vector_store_service

logger = structlog.get_logger(__name__)

# Lock đảm bảo chỉ 1 file chạy OCR tại 1 thời điểm (GPU không thread-safe)
_ocr_lock = asyncio.Lock()


class IngestionPipeline:
    """
    Orchestrator: file → text → chunks → vectors → Qdrant
    """

    def __init__(self):
        self.converter = document_converter
        self.splitter = ingestion_service
        self.vector_store = vector_store_service

    async def ingest_file(self, file_path: str, tenant_id: str, accessed_role: List[int], metadata: Optional[dict] = None) -> dict:
        """
        Pipeline chính:
        1. Convert file → text (hybrid: digital extract + OCR scan pages)
        2. Split chunks
        3. Upsert vào Qdrant (add_chunks tự embed dense + sparse)

        CPU-bound steps chạy trong thread pool để không block event loop.
        OCR steps serialize qua _ocr_lock để tránh GPU race condition.
        """
        path = Path(file_path)
        filename = path.name
        log = logger.bind(tenant_id=tenant_id, filename=filename, accessed_role=accessed_role)
        log.info("ingestion.start")

        loop = asyncio.get_event_loop()

        # --- Step 1: Extract text (chạy trong thread pool) ---
        async with _ocr_lock:
            raw_text = await loop.run_in_executor(
                None, self.converter.convert, path, True
            )

        if not raw_text or len(raw_text.strip()) < 10:
            log.warning("ingestion.empty_text")
            return {"status": "skipped", "reason": "empty content"}

        # --- Step 2: Split (CPU-bound → thread pool) ---
        chunks = await loop.run_in_executor(
            None,
            self.splitter.process_hybrid_splitting,
            raw_text,
            tenant_id,
            filename,
            accessed_role
        )
        log.info("ingestion.chunks_created", count=len(chunks))

        # --- Step 3: Xóa dữ liệu cũ (nếu re-upload) rồi upsert ---
        await loop.run_in_executor(
            None,
            self.vector_store.delete_document,
            tenant_id,
            filename
        )
        await loop.run_in_executor(
            None,
            self.vector_store.add_chunks,
            chunks,
        )

        log.info("ingestion.done", chunks=len(chunks))
        return {
            "status": "success",
            "filename": filename,
            "chunks": len(chunks),
        }

    async def delete_file(self, tenant_id: str, filename: str) -> dict:
        """Xóa chunks của 1 file khỏi Qdrant."""

        self.vector_store.delete_document(tenant_id=tenant_id, filename=filename)
        logger.info("ingestion.deleted", tenant_id=tenant_id, filename=filename)
        return {"status": "deleted", "deleted_count": 0}
