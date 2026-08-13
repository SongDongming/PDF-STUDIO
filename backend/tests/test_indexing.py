from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.services.assets import LocalObjectStorage, get_json, put_json
from app.services.indexing import (
    INDEX_DIMENSIONS,
    DocumentIndexer,
    IndexingError,
)
from app.services.providers import ProviderUnavailableError


def run(coroutine):
    return asyncio.run(coroutine)


class FakeEmbeddingProvider:
    provider_name = "fake"
    model = "document-model"
    dimensions = INDEX_DIMENSIONS
    index_signature = f"fake:document-model:{INDEX_DIMENSIONS}"

    def __init__(
        self, *, unavailable: bool = False, vector_dimensions: int = INDEX_DIMENSIONS
    ) -> None:
        self.unavailable = unavailable
        self.vector_dimensions = vector_dimensions
        self.document_batches: list[list[str]] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        if self.unavailable:
            raise ProviderUnavailableError("provider unavailable")
        return [
            [float(index == 0) for index in range(self.vector_dimensions)]
            for _ in texts
        ]

    async def embed_query(self, text: str) -> list[float]:
        return [float(index == 0) for index in range(self.vector_dimensions)]


@pytest.fixture
def compiled_storage(tmp_path: Path) -> tuple[LocalObjectStorage, str]:
    storage = LocalObjectStorage(tmp_path / "objects")
    prefix = "compiled/doc-1/v3"
    elements = [
        {
            "id": "el-text",
            "page": 1,
            "kind": "text",
            "bbox": {"x0": 100, "y0": 200, "x1": 900, "y1": 320},
            "bbox_normalized": [0.1, 0.2, 0.9, 0.32],
        },
        {
            "id": "el-figure",
            "page": 2,
            "kind": "figure",
            "asset_id": "asset-figure",
            "bbox": {"x0": 80, "y0": 300, "x1": 920, "y1": 1100},
            "bbox_normalized": [0.08, 0.3, 0.92, 0.8],
        },
    ]
    chunks = [
        {
            "id": "chunk-text",
            "document_id": "doc-1",
            "version": 3,
            "ordinal": 1,
            "markdown": "Agentic RAG 会根据证据充分度继续检索。",
            "page_start": 1,
            "page_end": 1,
            "element_ids": ["el-text"],
            "asset_ids": [],
            "heading_path": ["检索策略"],
        },
        {
            "id": "chunk-figure",
            "document_id": "doc-1",
            "version": 3,
            "ordinal": 2,
            "markdown": '<asset id="asset-figure" type="image">流程图',
            "page_start": 2,
            "page_end": 2,
            "element_ids": ["el-figure"],
            "asset_ids": ["asset-figure"],
            "heading_path": ["检索策略"],
        },
    ]
    elements_key = f"{prefix}/elements.json"
    chunks_key = f"{prefix}/chunks.json"
    manifest_key = f"{prefix}/manifest.json"
    put_json(
        storage,
        elements_key,
        {"document_id": "doc-1", "version": 3, "elements": elements},
    )
    put_json(
        storage,
        chunks_key,
        {"document_id": "doc-1", "version": 3, "chunks": chunks},
    )
    put_json(
        storage,
        manifest_key,
        {
            "schema_version": 1,
            "document_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "version": 3,
            "source": {
                "filename": "agentic-rag-handbook.pdf",
                "object_key": "source/kb-1/handbook.pdf",
            },
            "source_sha256": "a" * 64,
            "compiler_config_hash": "b" * 64,
            "chunk_count": 2,
            "chunks_key": chunks_key,
            "elements_key": elements_key,
            "assets": [
                {
                    "id": "asset-figure",
                    "document_id": "doc-1",
                    "version": 3,
                    "page": 2,
                    "element_id": "el-figure",
                    "kind": "figure",
                    "object_key": f"{prefix}/assets/asset-figure.png",
                    "source_page_key": f"{prefix}/pages/page-0002.png",
                    "bbox": {"x0": 80, "y0": 300, "x1": 920, "y1": 1100},
                    "bbox_normalized": [0.08, 0.3, 0.92, 0.8],
                }
            ],
        },
    )
    return storage, manifest_key


def test_builds_hybrid_index_in_document_mode_and_replays_idempotently(
    compiled_storage: tuple[LocalObjectStorage, str],
) -> None:
    storage, compiler_manifest_key = compiled_storage
    provider = FakeEmbeddingProvider()
    indexer = DocumentIndexer(
        storage=storage, embedder=provider, embedding_batch_size=1
    )

    result = run(
        indexer.index(
            compiler_manifest_key, document_title="Agentic RAG 图文手册"
        )
    )
    replay = run(
        indexer.index(
            compiler_manifest_key, document_title="Agentic RAG 图文手册"
        )
    )

    assert result.mode == "hybrid"
    assert result.dimensions == INDEX_DIMENSIONS
    assert result.index_signature == provider.index_signature
    assert result.chunk_count == 2
    assert storage.exists(result.payload_key)
    assert storage.exists(result.manifest_key)
    assert replay.idempotent_replay is True
    assert replay.manifest_key == result.manifest_key
    # One call per configured batch, and no calls during idempotent replay.
    assert provider.document_batches == [
        ["Agentic RAG 会根据证据充分度继续检索。"],
        ['<asset id="asset-figure" type="image">流程图'],
    ]

    payload = get_json(storage, result.payload_key)
    figure = next(item for item in payload["chunks"] if item["id"] == "chunk-figure")
    assert len(figure["embedding"]) == INDEX_DIMENSIONS
    assert figure["asset_ids"] == ["asset-figure"]
    assert figure["page"] == 2
    assert figure["bbox"] == [0.08, 0.3, 0.92, 0.8]
    assert figure["element_id"] == "el-figure"
    lineage = figure["metadata"]["asset_lineage"][0]
    assert lineage["object_key"].endswith("/asset-figure.png")
    assert lineage["source_page_key"].endswith("/page-0002.png")

    retriever = indexer.load_retriever(
        result.manifest_key, embedder=provider
    )
    restored = retriever.fetch_chunk("chunk-figure", knowledge_base_id="kb-1")
    assert restored is not None
    assert restored.asset_ids == ["asset-figure"]
    assert restored.element_id == "el-figure"


def test_provider_outage_persists_explicit_lexical_only_index(
    compiled_storage: tuple[LocalObjectStorage, str],
) -> None:
    storage, compiler_manifest_key = compiled_storage
    provider = FakeEmbeddingProvider(unavailable=True)
    indexer = DocumentIndexer(storage=storage, embedder=provider)

    result = run(indexer.index(compiler_manifest_key))

    assert result.mode == "lexical-only"
    assert result.dimensions is None
    assert result.index_signature == "lexical-only:v1"
    assert result.degraded_reason == "embedding_provider_unavailable"
    assert all(
        item["embedding"] is None
        for item in get_json(storage, result.payload_key)["chunks"]
    )
    retriever = indexer.load_retriever(result.manifest_key, embedder=provider)
    assert retriever.embedder is None
    hits = run(
        retriever.retrieve("证据检索", knowledge_base_id="kb-1")
    )
    assert hits


def test_wrong_embedding_dimensions_fail_closed_without_index_manifest(
    compiled_storage: tuple[LocalObjectStorage, str],
) -> None:
    storage, compiler_manifest_key = compiled_storage
    provider = FakeEmbeddingProvider(vector_dimensions=8)
    indexer = DocumentIndexer(storage=storage, embedder=provider)

    with pytest.raises(IndexingError) as caught:
        run(indexer.index(compiler_manifest_key))

    assert caught.value.code == "embedding_dimension_mismatch"
    index_manifests = [
        path
        for path in storage.root.rglob("manifest.json")
        if "indexes" in path.parts
    ]
    assert index_manifests == []


def test_dangling_multimodal_lineage_is_rejected(
    compiled_storage: tuple[LocalObjectStorage, str],
) -> None:
    storage, compiler_manifest_key = compiled_storage
    manifest = get_json(storage, compiler_manifest_key)
    manifest["assets"] = []
    put_json(storage, compiler_manifest_key, manifest)

    with pytest.raises(IndexingError) as caught:
        run(
            DocumentIndexer(
                storage=storage, embedder=FakeEmbeddingProvider()
            ).index(compiler_manifest_key)
        )

    assert caught.value.code == "invalid_asset_lineage"


def test_semantic_v2_overlay_enters_retrieval_text_and_index_identity(
    compiled_storage: tuple[LocalObjectStorage, str],
) -> None:
    storage, compiler_manifest_key = compiled_storage
    artifact = {
        "schema_version": 2,
        "semantic_artifact_version": "multimodal-semantic-v2",
        "document_id": "doc-1",
        "document_version": 3,
        "source_sha256": "a" * 64,
        "trusted_layout_contract": "paddle-v3-layout-lineage-v1",
        "page_enrichment_version": "deepseek-page-v1",
        "element_enrichment_version": "deepseek-element-v1",
        "pages": [
            {
                "page": 2,
                "publishable": True,
                "summary": "一张展示查询、图谱检索和图文回答的数据流。",
                "search_text": "知识图谱 多跳检索 图文回答",
                "element_relations": ["查询 -> 图谱 -> 回答"],
                "semantic_tags": ["知识图谱"],
            }
        ],
        "elements": [
            {
                "element_id": "el-figure",
                "publishable": True,
                "description": "蓝色流程图",
                "search_text": "Agentic RAG 调用图谱工具",
                "structure": {"relations": ["Agent -> Graph"]},
                "semantic_tags": ["Agentic RAG"],
            }
        ],
    }
    encoded = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    put_json(
        storage,
        "compiled/doc-1/v3/multimodal-semantic-v2.json",
        artifact,
    )
    provider = FakeEmbeddingProvider()
    result = run(
        DocumentIndexer(storage=storage, embedder=provider).index(
            compiler_manifest_key,
            document_title="Agentic RAG 图文手册",
        )
    )
    payload = get_json(storage, result.payload_key)
    figure = next(item for item in payload["chunks"] if item["id"] == "chunk-figure")

    assert "[DeepSeek整页视觉摘要]" in figure["text"]
    assert "Agentic RAG 调用图谱工具" in figure["text"]
    assert figure["metadata"]["semantic_enrichment_version"] == (
        "multimodal-semantic-v2"
    )
    assert (
        get_json(storage, result.manifest_key)["identity"]["semantic_fingerprint"]
        == artifact["artifact_sha256"]
    )
