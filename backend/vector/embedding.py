"""进程内密集向量服务；模型按需加载，并可在应用启动时预热。"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Callable, Literal

from langchain_huggingface import HuggingFaceEmbeddings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
EmbeddingState = Literal["uninitialized", "loading", "ready", "failed"]


@dataclass(frozen=True)
class EmbeddingStatus:
    state: EmbeddingState
    model: str
    device: str
    load_seconds: float | None = None
    error: str | None = None


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        },
    )
    embedder._client.max_seq_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "1024"))
    return embedder


class EmbeddingService:
    """线程安全的惰性单例包装；失败后不在每个请求里反复重试。"""

    def __init__(self, factory: Callable[[], HuggingFaceEmbeddings] = _create_dense_embedder):
        self._factory = factory
        self._embedder: HuggingFaceEmbeddings | None = None
        self._lock = threading.RLock()
        self._state: EmbeddingState = "uninitialized"
        self._load_seconds: float | None = None
        self._error: str | None = None
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.device = os.getenv("EMBEDDING_DEVICE", "cpu")

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    @property
    def status(self) -> dict[str, str | float | None]:
        with self._lock:
            return asdict(
                EmbeddingStatus(
                    state=self._state,
                    model=self.model_name,
                    device=self.device,
                    load_seconds=self._load_seconds,
                    error=self._error,
                )
            )

    def warmup(self) -> None:
        """加载权重并做一次最小推理；并发调用只会初始化一次。"""
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            if self._state == "failed":
                raise RuntimeError(f"嵌入模型此前加载失败: {self._error}")
            self._state = "loading"
            started = monotonic()
            try:
                embedder = self._factory()
                embedder.embed_query("warmup")
                self._embedder = embedder
                self._error = None
                self._state = "ready"
            except Exception as exc:
                self._embedder = None
                self._error = f"{type(exc).__name__}: {exc}"
                self._state = "failed"
                raise RuntimeError(f"嵌入模型加载失败: {exc}") from exc
            finally:
                self._load_seconds = round(monotonic() - started, 3)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.warmup()
        assert self._embedder is not None
        try:
            return self._embedder.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(f"本地嵌入模型调用失败: {exc}") from exc


# 全进程唯一实例；这里只创建轻量包装，不加载模型权重。
embedding_service = EmbeddingService()
