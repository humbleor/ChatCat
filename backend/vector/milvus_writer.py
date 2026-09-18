"""文档向量化并写入 Milvus - 稀疏向量由服务端 BM25 Function 自动生成"""

from backend.vector.embedding import EmbeddingService
from backend.vector.embedding import embedding_service as _default_embedding_service
from backend.vector.milvus_client import MilvusStore, get_milvus_store


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_store: MilvusStore = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_store = milvus_store or get_milvus_store()

    def write_documents(self, documents: list[dict], batch_size: int = 50, progress_callback=None):
        """
        批量写入文档到 Milvus（稀疏向量由 Milvus 的 BM25 Function 在插入时根据 text 自动计算）
        :param documents: 文档列表
        :param batch_size: 批次大小
        """
        if not documents:
            return

        self.milvus_store.init_collection()

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]

            # 只生成密集向量；sparse_embedding 字段不能手动提供（BM25 function 自动生成）
            dense_embeddings = self.embedding_service.get_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "text": doc["text"],
                    "chunk_uid": doc.get("chunk_uid") or doc["chunk_id"],
                    "document_id": doc.get("document_id", ""),
                    "is_deleted": bool(doc.get("is_deleted", False)),
                    "deleted_at": int(doc.get("deleted_at", 0) or 0),
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb in zip(batch, dense_embeddings)
            ]

            self.milvus_store.insert(insert_data)

            # 每个批次写入后更新进度，前端据此展示“向量化入库 xx%”。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
