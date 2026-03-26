import uuid
import shutil
import logging
import re
from pathlib import Path
from typing import List, Dict
from fastapi import UploadFile, HTTPException

from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.embedder import embedding_client
from transformers import AutoTokenizer
from ingestion.converter import document_converter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".json", ".csv"}

# TOKEN BASED
MIN_TOKENS = 200       # Nhỏ hơn mức này -> BẮT BUỘC GỘP (trừ khi đổi chủ đề)
IDEAL_TOKENS = 700     # Ok
MAX_TOKENS = 1200      # Lớn hơn mức này -> Semantic Split
HARD_CAP = 1500        # Ngưỡng cứng để cắt

class IngestionService:
    def __init__(self):
        # 1. Embedding Model & Tokenizer
        logger.info("⏳ Loading Embedding Model & Tokenizer...")

        self.embeddings = embedding_client.get_model()

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_client.model_name)
        logger.info("✅ Embedding Model & Tokenizer ready.")

        # 2. Semantic Splitter
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="gradient"
        )

        # 3. Markdown Splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
            ],
            strip_headers=False
        )

        # 4. Structure Splitter
        self.structure_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=0,
            separators=[
                "\n\n", "\n", 
                "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ", 
                "\na) ", "\nb) ", "\nc) ", "\nd) ",
                ". ",
            ]
        )

        # 5. Fallback Splitter
        self.fallback_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=HARD_CAP,
            chunk_overlap=100
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _preprocess_tables(self, text: str) -> str:
        """
        Tìm bảng Markdown và cô lập bằng \n\n để tránh bị cắt đôi.
        """
        # Regex: Tìm các dòng liên tiếp bắt đầu bằng | và chứa |
        # Group 1 bắt trọn vẹn khối bảng
        table_block_pattern = r'((?:^\|.*\|$\n?)+)'
        
        def isolate_table(match):
            table_content = match.group(1).strip()
            # Thêm \n bao quanh để RecursiveSplitter nhận diện đây là 1 khối
            return f"\n\n{table_content}\n\n"

        # Flags: MULTILINE để ^ và $ khớp đầu/cuối dòng
        return re.sub(table_block_pattern, isolate_table, text, flags=re.MULTILINE)

    def process_hybrid_splitting(self, text: str, tenant_id: str, filename: str, accessed_role: List[int]) -> List[Dict]:
        """
        Quy trình 5 lớp lọc:
        1. Pre-process Tables -> 2. MD Split -> 3. Structure Split -> 4. Smart Merge -> 5. Semantic/Hard Cap
        """
        # BƯỚC 1: Processing Tables
        text_safe_tables = self._preprocess_tables(text)

        # BƯỚC 2: Markdown Split
        raw_splits = self.header_splitter.split_text(text_safe_tables)
        
        # BƯỚC 3: Structure Refinement (Xử lý sơ bộ chunk quá to)
        refined_splits = []
        for doc in raw_splits:
            if self._count_tokens(doc.page_content) > MAX_TOKENS:
                sub_docs = self.structure_splitter.split_documents([doc])
                refined_splits.extend(sub_docs)
            else:
                refined_splits.append(doc)

        # BƯỚC 4: Smart Merge (Gộp các mảnh < MIN_TOKENS)
        merged_splits = self._smart_merge_sections(refined_splits)
        
        final_chunks = []

        # BƯỚC 5: Final Processing
        for doc in merged_splits:
            content = doc.page_content
            headers = doc.metadata
            token_count = self._count_tokens(content)

            # Case A: Chunk > MAX_TOKENS -> Dùng Semantic AI
            if token_count > MAX_TOKENS:
                logger.info(f"🔪 Semantic Splitting large chunk: {token_count} tokens")
                self._apply_semantic_split(final_chunks, content, headers, tenant_id, filename, accessed_role)

            # Case B: Chunk an toàn -> Lưu
            else:
                method = "markdown_adaptive" if token_count > MIN_TOKENS else "merged_small"
                self._add_chunk(final_chunks, content, headers, tenant_id, filename, accessed_role, method)

        return final_chunks

    def _smart_merge_sections(self, splits):
        """
        Logic Merge cập nhật:
        - Ưu tiên 1: Nếu < MIN_TOKENS -> BẮT BUỘC GỘP (trừ khi khác H1 hoặc vượt Hard Cap).
        - Ưu tiên 2: Nếu < IDEAL_TOKENS -> Gộp để tối ưu context.
        """
        if not splits: return []
        
        merged = []
        current_doc = splits[0]
        
        for next_doc in splits[1:]:
            curr_tokens = self._count_tokens(current_doc.page_content)
            next_tokens = self._count_tokens(next_doc.page_content)
            
            # Điều kiện kiểm tra Topic (H1)
            # KHÔNG gộp khác chương (H1) để tránh hallucination
            curr_h1 = current_doc.metadata.get("H1")
            next_h1 = next_doc.metadata.get("H1")
            same_topic = (curr_h1 == next_h1)

            # Điều kiện an toàn kích thước
            total_size = curr_tokens + next_tokens
            is_safe_size = total_size < HARD_CAP

            should_merge = False

            if same_topic and is_safe_size:
                # 1. Ép buộc merge nếu token (< 200)
                if curr_tokens < MIN_TOKENS:
                    should_merge = True
                # 2. Merge nếu tổng kích thước vẫn ok (< 700)
                elif total_size < IDEAL_TOKENS:
                    should_merge = True

            if should_merge:
                current_doc.page_content += "\n\n" + next_doc.page_content
                # Giữ metadata của chunk đầu tiên
            else:
                merged.append(current_doc)
                current_doc = next_doc
        
        merged.append(current_doc)
        return merged

    def _apply_semantic_split(self, chunks_list, content, headers, tenant_id, filename, accessed_role):
        try:
            sub_docs = self.semantic_splitter.create_documents([content])

            for sub in sub_docs:
                t_count = self._count_tokens(sub.page_content)

                if t_count > HARD_CAP:
                    logger.warning(f"⚠️ Semantic chunk ({t_count} tokens) > HARD_CAP. Forcing recursive split.")
                    hard_splits = self.fallback_splitter.split_text(sub.page_content)
                    for hard_txt in hard_splits:
                        self._add_chunk(chunks_list, hard_txt, headers, tenant_id, filename, accessed_role, "hard_cap_fallback")
                else:
                    self._add_chunk(chunks_list, sub.page_content, headers, tenant_id, filename, accessed_role, "hybrid_semantic")

        except Exception as e:
            logger.error(f"Semantic split error: {e}. Using Hard Cap Splitter.")
            hard_splits = self.fallback_splitter.split_text(content)
            for hard_txt in hard_splits:
                self._add_chunk(chunks_list, hard_txt, headers, tenant_id, filename, accessed_role, "hard_cap_error_fallback")

    def _add_chunk(self, chunks_list, content, headers, tenant_id, filename, accessed_role, method):
        enriched_content = self._inject_header_context(content, headers)

        flat_metadata = {
            "token_count": self._count_tokens(content),
            "method": method,
            "h1": headers.get("H1", ""),
            "h2": headers.get("H2", ""),
            "h3": headers.get("H3", "")
        }

        chunks_list.append({
            "chunk_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "accessed_role": accessed_role,
            "filename": filename,
            "content": enriched_content,
            "metadata": flat_metadata
        })

    def _inject_header_context(self, content: str, metadata: Dict) -> str:
        headers = [metadata[h] for h in ["H1", "H2", "H3"] if h in metadata]
        if not headers: return content
        context_str = " > ".join(headers)
        if context_str not in content[:300]: 
            return f"**Bối cảnh: {context_str}**\n\n{content}"
        return content

# Singleton instance
ingestion_service = IngestionService()

# Test
async def process_uploaded_file(file: UploadFile, tenant_id: str) -> List[Dict]:
    request_id = str(uuid.uuid4())
    # Tạo thư mục tạm riêng biệt cho request này
    temp_dir = Path(f"/tmp/ingest_{request_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Validate & Save file
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, "Định dạng không hỗ trợ.")
            
        temp_path = temp_dir / file.filename
        
        # Lưu file từ RAM xuống ổ cứng
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Convert to Markdown
        markdown_text = document_converter.convert(temp_path)
        
        # Xóa file ngay lập tức sau khi đã lấy được nội dung text
        # tránh đầy ổ cứng khi pipeline chunking phía sau chạy lâu
        if temp_path.exists():
            temp_path.unlink()
            # logger.info(f"🗑️ Đã xóa file tạm: {temp_path}")

        # 3. Chunking (Xử lý trên RAM)
        chunks = ingestion_service.process_hybrid_splitting(markdown_text, tenant_id, file.filename)
        
        logger.info(f"File: {file.filename} -> {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"❌ Lỗi xử lý file {file.filename}: {e}")
        raise e

    finally:
        # Dọn dẹp thư mục tạm (Safety net) luôn đảm bảo thư mục tạm được xóa sạch
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            # logger.info(f"🧹 Đã dọn dẹp thư mục tạm: {temp_dir}")