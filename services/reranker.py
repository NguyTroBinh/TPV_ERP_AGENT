import torch
import gc
import math
import logging
from typing import Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()

RERANKER_MODEL_NAME = "AITeamVN/Vietnamese_Reranker"

class RerankerService:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("reranker_loading", model=model_name, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()
        logger.info("reranker_ready", model=model_name)

    def rerank(self, query: str, documents: list[Any], top_k: int = 5, batch_size: int = 5) -> list[Any]:
        """
        Rerank danh sách documents theo relevance với query.

        Args:
            query:      Câu hỏi của user
            documents:  List kết quả từ Qdrant (ScoredPoint) hoặc dict {"content": ...} 
            top_k:      Số kết quả trả về sau rerank
            batch_size: Số cặp xử lý cùng lúc (5 = an toàn cho VRAM 12GB)

        Returns:
            List documents đã sort theo rerank score, cắt top_k
        """
        if not documents:
            return []

        # 1. Chuẩn bị pairs (query, content)
        pairs: list[list[str]] = []
        valid_docs: list[Any] = []

        for doc in documents:
            content = self._extract_content(doc)
            if content:
                pairs.append([query, content])
                valid_docs.append(doc)

        if not pairs:
            return []

        # 2. Batch inference
        all_scores: list[float] = []
        try:
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i + batch_size]

                    inputs = self.tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=1024,
                    ).to(self.device)

                    logits = self.model(**inputs, return_dict=True).logits
                    batch_scores = torch.sigmoid(logits.view(-1)).float()
                    all_scores.extend(batch_scores.cpu().tolist())

                    del inputs, logits, batch_scores

            # 3. Gán score và sort
            for doc, score in zip(valid_docs, all_scores):
                self._set_score(doc, score)

            valid_docs.sort(key=lambda d: self._get_score(d), reverse=True)

            logger.info(
                "rerank_done",
                query_len=len(query),
                input_docs=len(valid_docs),
                top_k=top_k,
            )
            return valid_docs[:top_k]

        except Exception as e:
            logger.error("rerank_error", error=str(e))
            return documents[:top_k]

        finally:
            if "all_scores" in locals():
                del all_scores
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    # ── Helpers — xử lý cả Qdrant ScoredPoint lẫn dict ──────

    @staticmethod
    def _extract_content(doc: Any) -> str:
        """Lấy content từ ScoredPoint (Qdrant) hoặc dict."""
        if isinstance(doc, dict):
            return doc.get("content", "")
        if hasattr(doc, "payload") and doc.payload:
            return doc.payload.get("content", "")
        return getattr(doc, "page_content", "")

    @staticmethod
    def _set_score(doc: Any, score: float) -> None:
        if isinstance(doc, dict):
            doc["rerank_score"] = score
        else:
            doc.score = score

    @staticmethod
    def _get_score(doc: Any) -> float:
        if isinstance(doc, dict):
            return doc.get("rerank_score", 0.0)
        return getattr(doc, "score", 0.0)


# Singleton
reranker_service = RerankerService()