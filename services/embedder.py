import logging
import torch
import gc
from typing import List
from sentence_transformers import SentenceTransformer

# Setup Logger
logger = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"

class VietnameseEmbeddingClient:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"⏳ Đang tải model Embedding: {model_name} lên {self.device.upper()}...")
        
        try:
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                trust_remote_code=True
            )
            
            # eval (không training) để tối ưu
            self.model.eval()
            
            logger.info("✅ Embedding Model đã sẵn sàng hoạt động!")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo Embedding Model: {e}")
            raise e

    def _clear_cache(self):
        """Dọn dẹp rác bộ nhớ sau khi tính toán xong."""

        if self.device == "cuda":
            gc.collect() 
            torch.cuda.empty_cache() 

    def get_embedding(self, text: str) -> List[float]:
        """
        Chuyển đổi 1 câu text (User Query) thành vector.
        """
        if not text: 
            return []
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Lỗi embed query: {e}")
            return []

    def embed_documents(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """
        Chuyển đổi danh sách văn bản (File Upload) thành vector.
        """
        if not texts: 
            return []
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=True, 
                convert_to_numpy=True,
                show_progress_bar=False 
            )
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Lỗi embed documents: {e}")
            return []
            
        finally:
            self._clear_cache()

    def encode(self, text: str) -> List[float]:
        """Alias cho get_embedding (để tương thích ngược nếu cần)"""

        return self.get_embedding(text)
    
    def get_model(self):
        return self.model

# Singleton Instance
embedding_client = VietnameseEmbeddingClient()