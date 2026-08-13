from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.assets import LocalObjectStorage, get_json
from app.services.enrichment import (
    ENRICHMENT_VERSION,
    EnrichmentBatchError,
    EnrichmentConfig,
    DeepSeekElementEnricher,
)
from app.services.ingestion import BoundingBox, NormalizedElement
from app.services.providers import ProviderUnavailableError


def element(
    element_id: str,
    kind: str,
    *,
    asset_key: str | None = None,
    page: int = 2,
) -> NormalizedElement:
    bbox = BoundingBox(10, 20, 110, 220)
    return NormalizedElement(
        id=element_id,
        page=page,
        order=1,
        kind=kind,
        label=kind,
        content=f"Paddle {kind} 原文",
        bbox=bbox,
        bbox_normalized=(0.1, 0.2, 0.5, 0.8),
        polygon=[(10, 20), (110, 20), (110, 220), (10, 220)],
        polygon_normalized=[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)],
        confidence=0.96,
        asset_id=f"asset-{element_id}" if asset_key else None,
        asset_key=asset_key,
    )


def provider_output(element_id: str, *, document_id: str = "doc-1") -> dict[str, Any]:
    return {
        "element_id": element_id,
        "document_id": document_id,
        "document_version": 3,
        "page": 2,
        "bbox": [0.1, 0.2, 0.5, 0.8],
        "description": "一张展示检索流程的架构图",
        "search_text": "Agentic RAG 检索、重排和回答流程",
        "structure": {
            "summary": "从查询到回答",
            "rows": [],
            "columns": [],
            "relations": ["查询 -> 检索", "检索 -> 回答"],
        },
        "semantic_tags": ["Agentic RAG", "检索"],
    }


class FakeProvider:
    def __init__(self, outputs: dict[str, dict[str, Any]] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = kwargs["user_text"]
        element_id = next(key for key in self.outputs if key in text)
        return self.outputs[element_id]


@pytest.fixture
def storage(tmp_path: Path) -> LocalObjectStorage:
    return LocalObjectStorage(tmp_path / "objects")


def test_enriches_only_multimodal_elements_with_data_url_and_strict_schema(
    storage: LocalObjectStorage,
) -> None:
    storage.put_bytes("assets/figure.png", b"\x89PNG\r\nfigure", "image/png")
    figure = element("el-figure", "figure", asset_key="assets/figure.png")
    text = element("el-text", "text")
    provider = FakeProvider({"el-figure": provider_output("el-figure")})

    result = asyncio.run(
        DeepSeekElementEnricher(storage=storage, provider=provider).enrich_elements(
            document_id="doc-1", document_version=3, elements=[text, figure]
        )
    )

    assert result.eligible_count == 1
    assert result.enriched_count == 1
    assert result.fully_enriched is True
    assert result.elements[0].publishable is True
    assert provider.calls[0]["images"][0].url.startswith("data:image/png;base64,")
    schema = provider.calls[0]["schema"]
    assert schema["additionalProperties"] is False
    assert {"description", "search_text", "structure", "semantic_tags"} <= set(
        schema["properties"]
    )


def test_versioned_cache_is_idempotent_and_bound_to_asset_bytes(
    storage: LocalObjectStorage,
) -> None:
    storage.put_bytes("assets/table.png", b"first", "image/png")
    table = element("el-table", "table", asset_key="assets/table.png")
    provider = FakeProvider({"el-table": provider_output("el-table")})
    enricher = DeepSeekElementEnricher(storage=storage, provider=provider)

    first = asyncio.run(
        enricher.enrich_elements(
            document_id="doc-1", document_version=3, elements=[table]
        )
    )
    second = asyncio.run(
        enricher.enrich_elements(
            document_id="doc-1", document_version=3, elements=[table]
        )
    )

    assert len(provider.calls) == 1
    assert first.elements[0].idempotent_replay is False
    assert second.elements[0].idempotent_replay is True
    cache_key = second.elements[0].cache_key
    assert cache_key == (
        f"enrichment/doc-1/v3/{ENRICHMENT_VERSION}/el-table.json"
    )
    assert get_json(storage, cache_key)["enrichment_version"] == ENRICHMENT_VERSION

    storage.put_bytes("assets/table.png", b"changed", "image/png")
    asyncio.run(
        enricher.enrich_elements(
            document_id="doc-1", document_version=3, elements=[table]
        )
    )
    assert len(provider.calls) == 2


def test_lineage_rewrite_degrades_and_is_never_fully_enriched(
    storage: LocalObjectStorage,
) -> None:
    storage.put_bytes("assets/formula.png", b"formula", "image/png")
    formula = element("el-formula", "formula", asset_key="assets/formula.png")
    rewritten = provider_output("el-formula", document_id="invented-doc")
    provider = FakeProvider({"el-formula": rewritten})

    result = asyncio.run(
        DeepSeekElementEnricher(storage=storage, provider=provider).enrich_elements(
            document_id="doc-1", document_version=3, elements=[formula]
        )
    )

    pending = result.elements[0]
    assert result.fully_enriched is False
    assert result.pending_count == 1
    assert pending.status == "pending_enrichment"
    assert pending.publishable is False
    assert pending.description == "Paddle 原始内容待增强"
    assert pending.error_code == "lineage_mismatch"
    assert pending.retryable is False
    assert pending.search_text == "Paddle formula 原文"


class FailingProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        raise self.error


def test_rate_limit_is_retryable_and_can_fail_closed(
    storage: LocalObjectStorage,
) -> None:
    storage.put_bytes("assets/figure.png", b"figure", "image/png")
    figure = element("el-figure", "figure", asset_key="assets/figure.png")
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(429, request=request)
    http_error = httpx.HTTPStatusError("limited", request=request, response=response)
    wrapped = ProviderUnavailableError("provider unavailable")
    wrapped.__cause__ = http_error

    degraded = asyncio.run(
        DeepSeekElementEnricher(
            storage=storage, provider=FailingProvider(wrapped)
        ).enrich_elements(
            document_id="doc-1", document_version=3, elements=[figure]
        )
    )
    assert degraded.elements[0].error_code == "rate_limited"
    assert degraded.elements[0].retryable is True
    assert degraded.fully_enriched is False

    with pytest.raises(EnrichmentBatchError) as caught:
        asyncio.run(
            DeepSeekElementEnricher(
                storage=storage,
                provider=FailingProvider(wrapped),
                config=EnrichmentConfig(allow_degraded=False),
            ).enrich_elements(
                document_id="doc-1", document_version=3, elements=[figure]
            )
        )
    assert caught.value.code == "rate_limited"
    assert caught.value.retryable is True


def test_missing_crop_remains_pending_and_non_publishable(
    storage: LocalObjectStorage,
) -> None:
    formula = element("el-formula", "formula")
    result = asyncio.run(
        DeepSeekElementEnricher(
            storage=storage,
            provider=FakeProvider({"el-formula": provider_output("el-formula")}),
        ).enrich_elements(
            document_id="doc-1", document_version=3, elements=[formula]
        )
    )
    assert result.elements[0].error_code == "asset_not_materialized"
    assert result.elements[0].retryable is False
    assert result.elements[0].publishable is False


class ConcurrencyProvider:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0.01)
            text = kwargs["user_text"]
            element_id = next(
                value for value in ("el-1", "el-2", "el-3") if value in text
            )
            output = provider_output(element_id)
            output["page"] = 2
            return output
        finally:
            self.active -= 1


def test_configured_maximum_concurrency_is_enforced(
    storage: LocalObjectStorage,
) -> None:
    elements = []
    for element_id in ("el-1", "el-2", "el-3"):
        key = f"assets/{element_id}.png"
        storage.put_bytes(key, element_id.encode(), "image/png")
        elements.append(element(element_id, "figure", asset_key=key))
    provider = ConcurrencyProvider()

    result = asyncio.run(
        DeepSeekElementEnricher(
            storage=storage,
            provider=provider,
            config=EnrichmentConfig(max_concurrency=2),
        ).enrich_elements(
            document_id="doc-1", document_version=3, elements=elements
        )
    )
    assert result.enriched_count == 3
    assert provider.maximum == 2


class PageProvider:
    def __init__(self, *, rewrite_page: bool = False) -> None:
        self.rewrite_page = rewrite_page
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "document_id": "doc-1",
            "document_version": 3,
            "page": 9 if self.rewrite_page else 2,
            "summary": "本页展示 Agentic RAG 的检索与回答流程。",
            "search_text": "查询 检索 知识图谱 视觉证据 回答",
            "section_intent": "解释系统的数据流",
            "element_relations": ["查询框 -> 检索器 -> 图文回答"],
            "semantic_tags": ["Agentic RAG", "流程图"],
        }


def test_page_level_k3_vision_preserves_trusted_paddle_lineage(
    storage: LocalObjectStorage,
) -> None:
    storage.put_bytes("pages/page-0002.png", b"page-image", "image/png")
    provider = PageProvider()
    result = asyncio.run(
        DeepSeekElementEnricher(storage=storage, provider=provider).enrich_pages(
            document_id="doc-1",
            document_version=3,
            pages=[
                {
                    "page": 2,
                    "width": 1000,
                    "height": 1400,
                    "image_key": "pages/page-0002.png",
                }
            ],
            elements=[element("el-figure", "figure")],
        )
    )

    assert result.fully_enriched is True
    assert result.pages[0].search_text.startswith("查询")
    assert provider.calls[0]["images"][0].url.startswith("data:image/png;base64,")
    assert "paddle_elements" in provider.calls[0]["user_text"]

    storage.put_bytes("pages/page-0002.png", b"changed-page-image", "image/png")
    rewritten = asyncio.run(
        DeepSeekElementEnricher(
            storage=storage, provider=PageProvider(rewrite_page=True)
        ).enrich_pages(
            document_id="doc-1",
            document_version=3,
            pages=[{"page": 2, "image_key": "pages/page-0002.png"}],
            elements=[element("el-figure", "figure")],
        )
    )
    assert rewritten.fully_enriched is False
    assert rewritten.pages[0].error_code == "lineage_mismatch"
