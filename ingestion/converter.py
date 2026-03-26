import re
import structlog
from pathlib import Path
from markitdown import MarkItDown

from ingestion.ocr_service import ocr

logger = structlog.get_logger(__name__)

class DocumentConverter:
    def __init__(self):
        self.office_converter = MarkItDown()

    def convert(self, file_path: Path, save_debug: bool = True) -> str:
            """
            Chuyển đổi file sang Markdown.
            Hỗ trợ: .pdf, .docx, .doc, .xlsx, .pptx, .json, .csv, .txt, .md
            """
            ext = file_path.suffix.lower()
            text = ""

            try:
                # 1: PDF — hybrid extraction (digital pages → pymupdf4llm, scan pages → OCR)
                if ext == ".pdf":
                    text = ocr.extract_hybrid(file_path)

                # 2: IMAGE — OCR toàn bộ
                elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                    text = ocr.extract_hybrid(file_path)

                # 3: TEXT & MARKDOWN (Đọc trực tiếp - KHÔNG qua Converter)
                elif ext in [".md", ".txt"]:
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        logger.warning("converter.encoding_fallback", file=file_path.name, fallback="latin-1")
                        text = file_path.read_text(encoding="latin-1")
                
                # 4: OFFICE & DATA (Dùng MarkItDown)
                elif ext in [".docx", ".doc", ".xlsx", ".pptx", ".json", ".csv"]:
                    # Kiểm tra xem file có tồn tại và hợp lệ không trước khi convert
                    if not file_path.exists():
                        raise FileNotFoundError(f"File không tồn tại: {file_path}")

                    result = self.office_converter.convert(str(file_path))
                    
                    # MarkItDown trả về object, cần lấy .text_content
                    if result and result.text_content:
                        text = result.text_content
                    else:
                        text = "" 
                
                else:
                    raise ValueError(f"Định dạng file chưa được hỗ trợ xử lý: {ext}")

                # XỬ LÝ HẬU KỲ
                if not text:
                    logger.warning("converter.empty_content", file=file_path.name)
                    return ""

                # 1. Làm sạch văn bản
                cleaned_text = self._cleanup_and_normalize(text)
                
                # 2. Lưu debug (nếu cần)
                if save_debug:
                    self._save_md_debug(file_path, cleaned_text)
                    
                return cleaned_text

            except Exception as e:
                logger.error("converter.error", file=file_path.name, error=str(e))
                raise RuntimeError(f"Lỗi convert file {file_path.name}: {str(e)}")

    def _save_md_debug(self, original_file_path: Path, content: str):
        """Lưu nội dung đã convert vào file .md cùng vị trí với file gốc."""
        try:
            # Tạo tên file: "ten_file.pdf" -> "ten_file.md"
            debug_path = original_file_path.with_suffix('.md')
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("converter.debug_saved", path=str(debug_path))
        except Exception as e:
            logger.warning("converter.debug_save_error", error=str(e))

    def _cleanup_and_normalize(self, text: str) -> str:
        """Làm sạch văn bản thô."""
        # Xử lý Header dính
        text = re.sub(r'([\.\;\n])\s*(\*\*[^*\n]{3,100}\*\*)', r'\1\n## \2', text)

        lines = text.split('\n')
        cleaned_lines = []
        universal_numbering = re.compile(r'^(\d+(\.\d+)*|[IVX]+|[A-Z]|[a-z])[\.\)\-]\s+')

        for line in lines:
            stripped = line.strip()
            if not stripped: continue
            if re.match(r'^[-_\s]*\d+[-_\s]*$', stripped): continue 
            if re.match(r'^[_=*-]{3,}$', stripped): continue 
            
            if len(stripped) < 200:
                if stripped.startswith('##'):
                    cleaned_lines.append(stripped.replace('**', ''))
                    continue
                if stripped.isupper() and len(stripped) > 4:
                    cleaned_lines.append(f"## {stripped}")
                    continue
                if stripped.startswith('**') and stripped.endswith('**'):
                    cleaned_lines.append(f"### {stripped.replace('**', '')}")
                    continue
                if universal_numbering.match(stripped):
                    cleaned_lines.append(f"### {stripped}")
                    continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

document_converter = DocumentConverter()