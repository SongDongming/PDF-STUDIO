"""Embedding providers with index-safe model and dimension contracts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import httpx

from app.services.providers import (
    AsyncHttpClient,
    ProviderConfigurationError,
    ProviderHealth,
    ProviderUnavailableError,
)


class EmbeddingProvider(Protocol):
    provider_name: str
    model: str
    dimensions: int

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def health(self) -> ProviderHealth: ...


class _HttpEmbeddingProvider:
    provider_name: str

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        client: AsyncHttpClient | None,
        timeout_seconds: float,
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError(
                f"{self.provider_name} embedding credential is not configured"
            )
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self._client = client
        self.timeout_seconds = timeout_seconds

    @property
    def index_signature(self) -> str:
        """Immutable signature persisted alongside every vector index."""

        return f"{self.provider_name}:{self.model}:{self.dimensions}"

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self.base_url}{path}",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=dict(payload),
                    timeout=self.timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}{path}",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=dict(payload),
                        timeout=self.timeout_seconds,
                    )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("embedding response is not an object")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError(
                f"{self.provider_name} embedding request failed"
            ) from exc

    async def health(self) -> ProviderHealth:
        try:
            vector = await self.embed_query("健康检查")
            healthy = len(vector) == self.dimensions
        except ProviderErrorTypes:
            healthy = False
        return ProviderHealth(
            provider=self.provider_name,
            model=self.model,
            configured=True,
            healthy=healthy,
            detail="模型服务可访问" if healthy else "模型服务暂不可用",
        )


ProviderErrorTypes = (ProviderUnavailableError, ValueError, KeyError, TypeError)


def _validate_vectors(
    vectors: Sequence[Sequence[Any]], *, expected_count: int, dimensions: int
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ProviderUnavailableError("embedding response count does not match input")
    normalized: list[list[float]] = []
    for vector in vectors:
        if len(vector) != dimensions:
            raise ProviderUnavailableError(
                "embedding response dimension does not match index contract"
            )
        try:
            normalized.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError(
                "embedding response contains a non-numeric value"
            ) from exc
    return normalized


class BailianEmbeddingProvider(_HttpEmbeddingProvider):
    """DashScope native ``text-embedding-v4`` adapter.

    The native endpoint is used instead of the OpenAI-compatible facade because
    it preserves the important ``query``/``document`` distinction.
    """

    provider_name = "aliyun-bailian"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com",
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        client: AsyncHttpClient | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            client=client,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_env(
        cls, *, client: AsyncHttpClient | None = None
    ) -> "BailianEmbeddingProvider":
        return cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com"
            ),
            model=os.environ.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1024")),
            client=client,
        )

    async def _embed(
        self, texts: Sequence[str], text_type: Literal["query", "document"]
    ) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        body = await self._post(
            "/api/v1/services/embeddings/text-embedding/text-embedding",
            {
                "model": self.model,
                "input": {"texts": list(texts)},
                "parameters": {
                    "text_type": text_type,
                    "dimension": self.dimensions,
                    "output_type": "dense",
                },
            },
        )
        try:
            items = sorted(body["output"]["embeddings"], key=lambda item: item["text_index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise ProviderUnavailableError(
                "Bailian returned an invalid embedding response"
            ) from exc
        return _validate_vectors(
            vectors, expected_count=len(texts), dimensions=self.dimensions
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "document")


class OpenAIEmbeddingProvider(_HttpEmbeddingProvider):
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-large",
        dimensions: int = 1024,
        client: AsyncHttpClient | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            client=client,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_env(
        cls, *, client: AsyncHttpClient | None = None
    ) -> "OpenAIEmbeddingProvider":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1024")),
            client=client,
        )

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        body = await self._post(
            "/embeddings",
            {
                "model": self.model,
                "input": list(texts),
                "dimensions": self.dimensions,
                "encoding_format": "float",
            },
        )
        try:
            items = sorted(body["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise ProviderUnavailableError(
                "OpenAI returned an invalid embedding response"
            ) from exc
        return _validate_vectors(
            vectors, expected_count=len(texts), dimensions=self.dimensions
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts)

