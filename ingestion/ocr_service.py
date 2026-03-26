import torch
import structlog
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import pymupdf4llm

logger = structlog.get_logger(__name__)

# Cấu hình
NAME_OCR_MODEL = "lightonai/LightOnOCR-2-1B"
# Số lượng ảnh xử lý cùng lúc.
# - VRAM 16GB trở lên: Set 4 hoặc 6
# - MPS (Mac): Set 1 hoặc 2 (MPS xử lý batch không hiệu quả bằng CUDA)
BATCH_SIZE = 4

# ── Ngưỡng phân loại trang ──
# Trang A4 văn bản thường > 500 ký tự, trang scan/ảnh < 50
CHAR_THRESHOLD_ABS = 100         # Tuyệt đối: < 100 ký tự → chắc chắn scan
CHAR_THRESHOLD_RATIO = 0.3       # Tương đối: < 30% trung bình các trang → nghi scan
MIN_FONTS_DIGITAL = 1            # Trang digital thường embed ít nhất 1 font


class OCR_Worker:
    def __init__(self, model_name=NAME_OCR_MODEL):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self._model = None
        self._processor = None
        self._model_name = model_name
        self.max_len = 2048

    def _ensure_model(self):
        """Lazy load model — chỉ load khi thực sự cần OCR."""
        if self._model is not None:
            return

        logger.info("ocr.loading_model", model=self._model_name, device=self.device)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self._processor = LightOnOcrProcessor.from_pretrained(self._model_name)
        self.max_len = getattr(self._model.config, "max_position_embeddings", 2048)
        logger.info("ocr.model_loaded", batch_size=BATCH_SIZE)

    @staticmethod
    def classify_pages(file_path: Path) -> list[bool]:
        """
        Phân loại từng trang PDF: True = cần OCR (scan/ảnh), False = digital (có text).

        Kết hợp 3 tín hiệu:
          1. Số ký tự tuyệt đối  — < CHAR_THRESHOLD_ABS → chắc chắn scan
          2. Số font embedded     — 0 font → scan (digital luôn embed font)
          3. Adaptive threshold   — ký tự < 30% trung bình → nghi scan

        Quyết định: cần ≥ 2/3 tín hiệu đồng ý "scan" → OCR trang đó.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return [True]

        if ext != ".pdf":
            return [False]

        try:
            pdf = pdfium.PdfDocument(str(path))
            num_pages = len(pdf)
            if num_pages == 0:
                pdf.close()
                return []

            # ── Pass 1: thu thập metrics từng trang ──
            page_metrics: list[dict] = []
            for i in range(num_pages):
                page = pdf[i]

                # Đếm ký tự
                textpage = page.get_textpage()
                text = textpage.get_text_bounded().strip()
                char_count = len(text)
                textpage.close()

                # Đếm font — page.get_objects() duyệt object, check type text
                font_count = 0
                seen_fonts: set[str] = set()
                for obj in page.get_objects():
                    if obj.type == pdfium.raw.FPDF_PAGEOBJ_TEXT:
                        try:
                            font_name = obj.get_font().name
                            if font_name and font_name not in seen_fonts:
                                seen_fonts.add(font_name)
                                font_count += 1
                        except Exception:
                            pass

                page_metrics.append({
                    "char_count": char_count,
                    "font_count": font_count,
                })

            pdf.close()

            # ── Pass 2: tính adaptive threshold ──
            char_counts = [m["char_count"] for m in page_metrics]
            avg_chars = sum(char_counts) / num_pages if num_pages > 0 else 0
            adaptive_threshold = avg_chars * CHAR_THRESHOLD_RATIO

            # ── Pass 3: vote — cần ≥ 2/3 tín hiệu ──
            needs_ocr: list[bool] = []
            for i, m in enumerate(page_metrics):
                votes = 0

                # Signal 1: quá ít ký tự
                if m["char_count"] < CHAR_THRESHOLD_ABS:
                    votes += 1

                # Signal 2: không có font embedded
                if m["font_count"] < MIN_FONTS_DIGITAL:
                    votes += 1

                # Signal 3: ký tự thấp hơn nhiều so với trung bình
                if avg_chars > 0 and m["char_count"] < adaptive_threshold:
                    votes += 1

                needs_ocr.append(votes >= 2)

            ocr_count = sum(needs_ocr)
            logger.info(
                "ocr.classify_pages",
                file=path.name,
                total_pages=num_pages,
                ocr_pages=ocr_count,
                digital_pages=num_pages - ocr_count,
                avg_chars=round(avg_chars),
                adaptive_threshold=round(adaptive_threshold),
            )
            return needs_ocr

        except Exception as e:
            logger.warning("ocr.classify_error", file=path.name, error=str(e))
            return [True]  # Lỗi → OCR cho an toàn

    @staticmethod
    def is_file_scanned(file_path: Path) -> bool:
        """Compat: True nếu file có BẤT KỲ trang nào cần OCR."""
        return any(OCR_Worker.classify_pages(file_path))

    def extract_hybrid(self, file_path: Path) -> str:
        """
        Hybrid extraction — per-page strategy:
        - Trang digital (có text) → pymupdf4llm extract (nhanh, chính xác)
        - Trang scan (ít/không text) → OCR model (chậm hơn, cần GPU)

        Trả về markdown text đã ghép từ tất cả các trang.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # File ảnh → OCR toàn bộ
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return self.extract_text_content(file_path)

        if ext != ".pdf":
            return ""

        page_classification = self.classify_pages(file_path)
        if not page_classification:
            return ""

        total_pages = len(page_classification)
        ocr_pages = [i for i, need_ocr in enumerate(page_classification) if need_ocr]
        digital_pages = [i for i, need_ocr in enumerate(page_classification) if not need_ocr]

        logger.info(
            "ocr.hybrid_start",
            file=file_path.name,
            total=total_pages,
            ocr_pages=len(ocr_pages),
            digital_pages=len(digital_pages),
        )

        # Kết quả theo thứ tự trang
        page_texts: list[str] = [""] * total_pages

        # ── 1. Digital pages: pymupdf4llm (nhanh) ──
        if digital_pages:
            try:
                all_md = pymupdf4llm.to_markdown(
                    str(file_path),
                    pages=digital_pages,
                )
                # pymupdf4llm trả về 1 string cho tất cả pages được chọn
                # Split theo form feed hoặc page break
                chunks = all_md.split("\f") if "\f" in all_md else [all_md]

                for idx, page_num in enumerate(digital_pages):
                    if idx < len(chunks):
                        page_texts[page_num] = chunks[idx].strip()

                logger.info("ocr.digital_extracted", pages=len(digital_pages))
            except Exception as e:
                logger.warning("ocr.digital_error", error=str(e))

        # ── 2. OCR pages: model inference (chậm) ──
        if ocr_pages:
            self._ensure_model()
            ocr_results = self._ocr_pages(file_path, ocr_pages)
            for page_num, text in zip(ocr_pages, ocr_results):
                page_texts[page_num] = text

            logger.info("ocr.ocr_extracted", pages=len(ocr_pages))

        result = "\n\n---\n\n".join(t for t in page_texts if t)
        logger.info("ocr.hybrid_done", file=file_path.name, total_chars=len(result))
        return result

    def _ocr_pages(self, pdf_path: Path, page_numbers: list[int]) -> list[str]:
        """OCR các trang cụ thể của PDF."""
        pil_images = []
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            for page_num in page_numbers:
                img = pdf[page_num].render(scale=3.0).to_pil()
                pil_images.append(img)
            pdf.close()
        except Exception as e:
            logger.error("ocr.render_error", error=str(e))
            return [""] * len(page_numbers)

        # Batch inference
        results = []
        total = len(pil_images)
        for i in range(0, total, BATCH_SIZE):
            batch = pil_images[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info("ocr.batch_processing", batch=f"{batch_num}/{total_batches}", images=len(batch))

            try:
                batch_texts = self._process_batch(batch)
                results.extend(batch_texts)
            except Exception as e:
                logger.error("ocr.batch_error", batch=batch_num, error=str(e))
                results.extend([""] * len(batch))

            # Dọn VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Dọn RAM
        for img in pil_images:
            img.close()

        return results

    def _process_batch(self, image_batch: list[Image.Image]) -> list[str]:
        """Xử lý một batch ảnh cùng lúc (Batch Inference)."""
        if not image_batch:
            return []

        self._ensure_model()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract all text from this document and convert to markdown format."},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch_prompts = [text_prompt] * len(image_batch)

        inputs = self._processor(
            text=batch_prompts,
            images=image_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = max(self.max_len - input_len - 50, 1024)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )

        generated_text_batch = self._processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text_batch

    def extract_text_content(self, input_path: Path) -> str:
        """
        OCR toàn bộ file (ảnh hoặc PDF).
        Dùng cho file ảnh đơn hoặc khi muốn force OCR toàn bộ.
        """
        input_path = Path(input_path)
        self._ensure_model()

        pil_images = []
        try:
            if input_path.suffix.lower() == ".pdf":
                pdf = pdfium.PdfDocument(input_path)
                total_pages = len(pdf)
                logger.info("ocr.full_start", file=input_path.name, pages=total_pages)
                for i in range(total_pages):
                    img = pdf[i].render(scale=3.0).to_pil()
                    pil_images.append(img)
                pdf.close()
            else:
                with Image.open(input_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    pil_images.append(img.copy())

            results = []
            total = len(pil_images)
            for i in range(0, total, BATCH_SIZE):
                batch = pil_images[i:i + BATCH_SIZE]
                try:
                    batch_texts = self._process_batch(batch)
                    results.extend(batch_texts)
                except Exception as e:
                    logger.error("ocr.batch_error", batch=i, error=str(e))
                    results.append(f"[Error extracting page {i}]")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            logger.info("ocr.full_done", file=input_path.name, pages=total)
            return "\n\n---\n\n".join(results)

        except Exception as e:
            logger.error("ocr.critical_error", file=input_path.name, error=str(e))
            return ""
        finally:
            for img in pil_images:
                img.close()


# Singleton — lazy load, model chỉ load khi gọi OCR lần đầu
ocr = OCR_Worker()
