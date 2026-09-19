"""Reranker provider adapters for Jina-compatible APIs and Hugging Face TEI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import requests


class RerankerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


@dataclass(frozen=True)
class RerankerSettings:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float

    @property
    def enabled(self) -> bool:
        if not self.model or not self.base_url:
            return False
        return self.provider != "jina" or bool(self.api_key)

    @property
    def endpoint(self) -> str:
        base = self.base_url.strip().rstrip("/")
        if not base:
            return ""
        if self.provider == "tei":
            return base if base.endswith("/rerank") else f"{base}/rerank"
        return base if base.endswith("/v1/rerank") else f"{base}/v1/rerank"


def settings_from_env() -> RerankerSettings:
    provider = os.getenv("RERANK_PROVIDER", "jina").strip().lower()
    if provider not in {"jina", "tei"}:
        raise ValueError(f"不支持的 RERANK_PROVIDER: {provider}")
    return RerankerSettings(
        provider=provider,
        model=(os.getenv("RERANK_MODEL") or "").strip(),
        base_url=(os.getenv("RERANK_BINDING_HOST") or "").strip(),
        api_key=(os.getenv("RERANK_API_KEY") or "").strip(),
        timeout_seconds=float(os.getenv("RERANK_TIMEOUT_SECONDS", "15")),
    )


class RerankerClient:
    def __init__(
        self,
        settings: RerankerSettings,
        *,
        post: Callable[..., Any] = requests.post,
    ):
        self.settings = settings
        self._post = post

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _payload(self, query: str, texts: list[str], top_n: int) -> dict[str, Any]:
        if self.settings.provider == "tei":
            return {
                "query": query,
                "texts": texts,
                "truncate": True,
                "raw_scores": False,
                "return_text": False,
            }
        return {
            "model": self.settings.model,
            "query": query,
            "documents": texts,
            "top_n": min(top_n, len(texts)),
            "return_documents": False,
        }

    def _parse(self, data: Any, top_n: int) -> list[RerankItem]:
        if self.settings.provider == "tei":
            # TEI HTTP currently returns a rank list. Also accept the gRPC-shaped
            # {"ranks": [...]} representation so upgrades do not break the adapter.
            items = data.get("ranks", []) if isinstance(data, dict) else data
            score_key = "score"
        else:
            items = data.get("results", []) if isinstance(data, dict) else []
            score_key = "relevance_score"

        if not isinstance(items, list):
            raise RerankerError("reranker response items must be a list")

        parsed: list[RerankItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get(score_key)
            if isinstance(index, int) and isinstance(score, (int, float)):
                parsed.append(RerankItem(index=index, score=float(score)))
        parsed.sort(key=lambda item: item.score, reverse=True)
        return parsed[:top_n]

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[RerankItem]:
        if not self.settings.enabled:
            return []
        response = self._post(
            self.settings.endpoint,
            headers=self._headers(),
            json=self._payload(query, texts, top_n),
            timeout=self.settings.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RerankerError(f"HTTP {response.status_code}: {response.text}")
        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise RerankerError(f"reranker returned invalid JSON: {exc}") from exc
        items = self._parse(data, min(top_n, len(texts)))
        if not items:
            raise RerankerError("empty_rerank_results")
        return items


reranker_client = RerankerClient(settings_from_env())
