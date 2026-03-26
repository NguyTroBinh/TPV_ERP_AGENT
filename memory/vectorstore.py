import structlog
import uuid
import math
import hashlib
import gc
import torch
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Optional
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from services.embedder import embedding_client

# Thread pool dùng chung — song song hóa dense + sparse embedding
_embed_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")

logger = structlog.get_logger(__name__)

from config import get_settings as _get_settings
_vs_settings = _get_settings()

# Setup DB
QDRANT_URL = _vs_settings.vector_store.qdrant_url
QDRANT_API_KEY = _vs_settings.vector_store.qdrant_api_key or None
ENTERPRISE_COLL_NAME = "chatbot_knowledge_for_businesses"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse" 
DENSE_DIMENSION = 1024 

SPARSE_MODEL_NAME = "Qdrant/bm25"

class VectorStoreService:
    def __init__(self, shard_number: int = 2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Connect to Qdrant
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) 
        self.collection_name = ENTERPRISE_COLL_NAME
        self.vector_size = DENSE_DIMENSION
        self.dense_vector = DENSE_VECTOR_NAME
        self.sparse_vector = SPARSE_VECTOR_NAME
        self.shard_number = shard_number
        
        # 2. Load Model Sparse (BM25)
        logger.info(f"⏳ Đang tải model Sparse (BM25) trên thiết bị: {self.device}...")
        try:
            # providers=["CUDAExecutionProvider"] nếu muốn ép GPU, hoặc auto
            self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
            logger.info("✅ Model Sparse đã sẵn sàng.")
        except Exception as e:
            logger.error(f"❌ Lỗi tải Model Sparse: {e}")
            raise e

        # 3. Load Collection
        self._ensure_collection()

    def _clear_cache(self):
        """Dọn dẹp bộ nhớ chung cho cả class"""

        # Clear VRAM
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _ensure_collection(self):
        """Tạo collection hỗ trợ Dense + Sparse vector, tối ưu cho production."""

        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Đang tạo collection '{self.collection_name}'...")
            # Create Collection
            self.client.create_collection(
                collection_name=self.collection_name,

                # Create Dense Vector
                vectors_config={
                    self.dense_vector: models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                        # Scalar Quantization: float32 → int8
                        # Giảm ~4x RAM, search nhanh hơn, accuracy giảm ~1%
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                # Rescore từ original vector để giữ accuracy
                                always_ram=True,
                            ),
                        ),
                    )
                },
                # Create Sparse Vector
                sparse_vectors_config={
                    self.sparse_vector: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                },
                # Tắt HNSW để tăng tốc độ upload
                hnsw_config=models.HnswConfigDiff(m=0),
                # Tối ưu segment merging
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    default_segment_number=2,
                ),
                shard_number=self.shard_number,
            )

            # Create Payload Indexes for tenant_id, filename and accessed_role fields 
            try:
                self.client.create_payload_index(self.collection_name, "tenant_id", models.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(self.collection_name, "filename", models.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(self.collection_name, "accessed_role", models.PayloadSchemaType.INTEGER)
            except Exception:
                pass

            logger.info("Collection đã tạo xong.")

    def optimize_indexing(self):
        """Bật lại Indexing sau khi upload xong để tìm kiếm nhanh hơn."""

        logger.info("Đang bật HNSW indexing...")
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,               # Số connections/node — 16 cân bằng tốt
                ef_construct=128,   # Chất lượng build index (cao hơn = chính xác hơn, build chậm hơn)
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        logger.info("HNSW indexing đã bật.")

    # Generate deterministic ID
    def generate_deterministic_id(self, tenant_id: str, filename: str, chunk_idx: int) -> str:
        unique_str = f"{tenant_id}_{filename}_{chunk_idx}"
        hash_obj = hashlib.md5(unique_str.encode('utf-8'))
        return str(uuid.UUID(hash_obj.hexdigest()))

    # Delete document by tenant_id and filename
    def delete_document(self, tenant_id: str, filename: str):
        """Xóa toàn bộ chunks của một file cụ thể dựa trên tenant_id và filename."""

        must = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key="filename", match=models.MatchValue(value=filename)),
        ]
        filter_condition = models.Filter(must=must)

        count = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_condition,
            exact=True,
        ).count

        if count > 0:
            logger.info(f"🗑️ Xóa dữ liệu: {filename} ({count} chunks, tenant={tenant_id})")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_condition)
            )

    # Add batch chunks + Clear Cache
    def add_chunks(self, chunks: List[Dict], batch_size: int = 256):
        """
        Upload chunks với cơ chế dọn Cache tích cực.
        """
        if not chunks: 
            return

        total_chunks = len(chunks)
        total_batches = math.ceil(total_chunks / batch_size)
        
        logger.info(f"💾 Bắt đầu upload {total_chunks} chunks ({total_batches} batches)...")

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            current_batch_idx = (i // batch_size) + 1
            
            try:
                # 1. Lấy text
                texts = [chunk['content'] for chunk in batch_chunks]

                # 2. Tạo Dense + Sparse vectors SONG SONG (2 model độc lập)
                dense_future = _embed_pool.submit(embedding_client.embed_documents, texts)
                sparse_future = _embed_pool.submit(lambda t: list(self.sparse_model.embed(t)), texts)

                dense_vectors = dense_future.result()
                sparse_vectors = sparse_future.result()
                
                points = []
                for j, chunk in enumerate(batch_chunks):                   
                    payload = {
                        "tenant_id": chunk.get("tenant_id"),
                        "filename": chunk.get("filename"),
                        "accessed_role": chunk.get("accessed_role"),
                        "content": chunk.get("content"),
                        "metadata": chunk.get("metadata", {})
                    }

                    # Tạo ID cố định
                    global_idx = i + j
                    point_id = self.generate_deterministic_id(
                        payload["tenant_id"],
                        payload["filename"],
                        global_idx
                    )

                    # Format vectors
                    dense_vec = dense_vectors[j]
                    if hasattr(dense_vec, 'tolist'): dense_vec = dense_vec.tolist()

                    raw_sparse = sparse_vectors[j]
                    sparse_vec = models.SparseVector(
                        indices=raw_sparse.indices.tolist(),
                        values=raw_sparse.values.tolist()
                    )

                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            self.dense_vector: dense_vec,
                            self.sparse_vector: sparse_vec
                        },
                        payload=payload
                    ))

                # 4. Upsert Batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.info(f"   Using Batch {current_batch_idx}/{total_batches}: Đã nạp {len(points)} chunks.")

            except Exception as e:
                logger.error(f"❌ Lỗi tại Batch {current_batch_idx}: {e}")
                raise e
            
            finally:
                # Clear memory 
                if 'dense_vectors' in locals(): del dense_vectors
                if 'sparse_vectors' in locals(): del sparse_vectors
                if 'points' in locals(): del points
                self._clear_cache()
        
        logger.info("✅ Hoàn tất toàn bộ quá trình upload.")

    def search_hybrid(self, query: str, tenant_id: str, filenames: Optional[List[str]] = None, role_id: Optional[int] = None, k: int = 20, top_k: Optional[int] = None):
        if top_k is not None: 
            k = top_k

        try:
            # 1+2. Dense + Sparse embedding song song
            dense_future = _embed_pool.submit(embedding_client.get_embedding, query)
            sparse_future = _embed_pool.submit(lambda q: list(self.sparse_model.embed(q))[0], query)

            dense_vector = dense_future.result()
            sparse_emb = sparse_future.result()
            sparse_vector = models.SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )

            # 3. Setup Prefetch — filter theo tenant_id + filenames + role_id
            prefetch_limit = k

            must_conditions = []
            if tenant_id: must_conditions.append(models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)))
            if filenames is not None: must_conditions.append(models.FieldCondition(key="filename", match=models.MatchAny(any=filenames)))
            if role_id is not None: must_conditions.append(models.FieldCondition(key="accessed_role", match=models.MatchValue(value=role_id)))
            
            filter_condition = models.Filter(must=must_conditions)

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=sparse_vector, using=self.sparse_vector, limit=prefetch_limit, filter=filter_condition),
                    models.Prefetch(query=dense_vector, using=self.dense_vector, limit=prefetch_limit, filter=filter_condition),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
                search_params=models.SearchParams(
                    hnsw_ef=128,        # Search accuracy (cao hơn = chính xác hơn, chậm hơn)
                    quantization=models.QuantizationSearchParams(
                        rescore=True,   # Rescore top candidates từ original vectors → giữ accuracy
                    ),
                ),
            )

            return results.points
        except Exception as e:
            logger.error(f"Search Error: {e}")
            return []
    
    def health_check(self) -> bool:
        """Kiểm tra kết nối Qdrant — dùng trong health endpoint."""

        collections = self.client.get_collections()
        return collections is not None

# Singleton
vector_store_service = VectorStoreService()