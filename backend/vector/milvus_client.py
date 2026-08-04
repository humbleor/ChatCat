"""Milvus 访问层：无状态 Store + 进程级单例 gRPC 连接（closed channel 自动重建）。

稀疏向量由 Milvus 服务端 BM25 Function 在插入时根据 text 字段自动生成，
客户端不再计算/上传 sparse_embedding，也不再维护 data/bm25_state.json 那套状态。
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

load_dotenv()

QUERY_MAX_LIMIT = 16384
T = TypeVar("T")


@dataclass(frozen=True)
class MilvusSettings:
    host: str
    port: str
    collection_name: str
    uri: str
    timeout: float

    @classmethod
    def from_env(cls) -> MilvusSettings:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        timeout = float(os.getenv("MILVUS_TIMEOUT", "30"))
        return cls(
            host=host,
            port=port,
            collection_name=collection,
            uri=f"http://{host}:{port}",
            timeout=timeout,
        )


def _normalize_filter(filter_expr: str) -> str:
    return filter_expr.strip() if filter_expr.strip() else "id >= 0"


def _ensure_no_proxy_localhost() -> None:
    # 确保 gRPC 直连本地 Milvus，不被 http(s)_proxy 劫持。

    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        missing = [host for host in ("127.0.0.1", "localhost") if host not in parts]
        if not missing:
            continue
        os.environ[var] = (current.rstrip(",") + "," if current else "") + ",".join(missing)


class MilvusStore:
    """Milvus 集合读写；对外无状态，底层复用一个进程级 gRPC 连接。

    连接采用懒创建 + 进程内单例：遇到 closed-channel 类错误时自动丢弃并重建重试一次，
    避免长时间持有失效 channel，又不为每个操作付出建连开销。
    """

    def __init__(self, settings: MilvusSettings | None = None):
        self._settings = settings or MilvusSettings.from_env()
        self._client: MilvusClient | None = None
        self._lock = threading.RLock()

    @property
    def collection_name(self) -> str:
        return self._settings.collection_name

    # ---- 连接管理（单例 + 失效重建） ----

    def _get_client(self) -> MilvusClient:
        with self._lock:
            if self._client is None:
                _ensure_no_proxy_localhost()
                self._client = MilvusClient(uri=self._settings.uri, timeout=self._settings.timeout)
            return self._client

    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "closed channel" in str(exc).lower()

    @staticmethod
    def _close_client(client: MilvusClient) -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    def _reset_client(self, failed_client: MilvusClient | None = None) -> None:
        with self._lock:
            if self._client is None:
                return
            if failed_client is not None and self._client is not failed_client:
                return
            client = self._client
            self._client = None
        self._close_client(client)

    def _run(self, operation: Callable[[MilvusClient], T]) -> T:
        client = self._get_client()
        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise
            self._reset_client(client)
            return operation(self._get_client())

    @contextmanager
    def session(self) -> Iterator[MilvusClient]:
        """标记一次业务流（如整次上传）；连接为进程级共享，进出不再单独创建/关闭。"""
        yield self._get_client()

    # ---- 集合与数据操作 ----

    @staticmethod
    def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int) -> None:
        if client.has_collection(collection_name):
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
            enable_match=True,
        )
        schema.add_field("filename", DataType.VARCHAR, max_length=255)
        schema.add_field("file_type", DataType.VARCHAR, max_length=50)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_level", DataType.INT64)

        # 服务端 BM25 Function：插入时根据 text 自动生成 sparse_embedding
        bm25_function = Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_embedding"],
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 256},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def init_collection(self, dense_dim: int | None = None) -> None:
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        def _init(client: MilvusClient) -> None:
            self.ensure_collection(client, self.collection_name, dense_dim)

        self._run(_init)

    def insert(self, data: list[dict]):
        return self._run(lambda client: client.insert(self.collection_name, data))

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        expr = _normalize_filter(filter_expr)
        fields = output_fields or ["filename", "file_type"]

        def _query(client: MilvusClient):
            return client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )

        return self._run(_query)

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """分页拉取；单次 session 内完成，避免每页重建连接。"""
        fields = output_fields or ["filename", "file_type"]
        expr = _normalize_filter(filter_expr)

        def _query_all(client: MilvusClient) -> list:
            out: list = []
            offset = 0
            while True:
                batch = client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < QUERY_MAX_LIMIT:
                    break
                offset += len(batch)
            return out

        return self._run(_query_all)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join(f'"{item}"' for item in ids)
        return self.query(
            filter_expr=f"chunk_id in [{quoted_ids}]",
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        # 稀疏向量由服务端 BM25 Function 从 query 文本实时生成
        sparse_search = AnnSearchRequest(
            data=[query],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        reranker = RRFRanker(k=rrf_k)

        def _search(client: MilvusClient):
            return client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": hit.get("text", ""),
                        "filename": hit.get("filename", ""),
                        "file_type": hit.get("file_type", ""),
                        "page_number": hit.get("page_number", 0),
                        "chunk_id": hit.get("chunk_id", ""),
                        "parent_chunk_id": hit.get("parent_chunk_id", ""),
                        "root_chunk_id": hit.get("root_chunk_id", ""),
                        "chunk_level": hit.get("chunk_level", 0),
                        "chunk_idx": hit.get("chunk_idx", 0),
                        "score": hit.get("distance", 0.0),
                    }
                )
        return formatted_results

    def dense_retrieve(
        self,
        dense_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": hit.get("entity", {}).get("text", ""),
                        "filename": hit.get("entity", {}).get("filename", ""),
                        "file_type": hit.get("entity", {}).get("file_type", ""),
                        "page_number": hit.get("entity", {}).get("page_number", 0),
                        "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                        "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                        "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                        "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                        "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                        "score": hit.get("distance", 0.0),
                    }
                )
        return formatted_results

    def delete(self, filter_expr: str):
        return self._run(lambda client: client.delete(collection_name=self.collection_name, filter=filter_expr))

    def has_collection(self) -> bool:
        return self._run(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self) -> None:
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run(_drop)


_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
