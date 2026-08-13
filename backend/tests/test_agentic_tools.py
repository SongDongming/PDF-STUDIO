from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schemas import RagSettings
from app.services.agentic_tools import (
    AgentRequestScope,
    FetchEvidenceTool,
    InspectVisualTool,
    SearchChunksTool,
    SearchGraphTool,
    bind_agent_request,
)
from app.services.assets import LocalObjectStorage
from app.services.graph_repository import (
    EvidenceLineage,
    GraphEdgeRecord,
    GraphNodeRecord,
    InMemoryGraphRepository,
)
from app.services.retrieval import HybridRetriever, RetrievalChunk


class ToolStore(SimpleNamespace):
    async def retrieve(
        self,
        *,
        query: str,
        knowledge_base_id: str,
        rag: RagSettings,
        document_ids: list[str],
    ):
        hits = await self.retriever.retrieve(
            query, knowledge_base_id=knowledge_base_id, settings=rag
        )
        if document_ids:
            hits = [hit for hit in hits if hit.chunk.document_id in document_ids]
        return hits


@pytest.fixture
def tool_store(tmp_path: Path) -> ToolStore:
    storage = LocalObjectStorage(tmp_path / "objects")
    storage.put_bytes("assets/figure.png", b"visual-pixels", "image/png")
    retriever = HybridRetriever(
        [
            RetrievalChunk(
                id="chunk-1",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                document_title="多模态检索手册",
                page=4,
                text="Kimi K3 与 PaddleOCR 协同完成图文 PDF 检索。",
                bbox=(0.1, 0.2, 0.9, 0.8),
                element_id="element-1",
                asset_ids=["asset-1"],
            ),
            RetrievalChunk(
                id="chunk-other",
                knowledge_base_id="kb-2",
                document_id="doc-2",
                document_title="隔离知识库",
                page=1,
                text="Kimi K3 与 PaddleOCR",
            ),
        ]
    )
    graph = InMemoryGraphRepository()
    graph.commit_document(
        "kb-1",
        "doc-1",
        [
            GraphNodeRecord(
                id="entity:kimi",
                knowledge_base_id="kb-1",
                kind="entity",
                label="Kimi K3",
            ),
            GraphNodeRecord(
                id="entity:paddle",
                knowledge_base_id="kb-1",
                kind="entity",
                label="PaddleOCR",
            ),
        ],
        [
            GraphEdgeRecord(
                id="relation:works-with",
                knowledge_base_id="kb-1",
                source="entity:kimi",
                target="entity:paddle",
                relation="WORKS_WITH",
                evidence=(
                    EvidenceLineage(
                        document_id="doc-1",
                        page=4,
                        bbox=(0.1, 0.2, 0.9, 0.8),
                        chunk_id="chunk-1",
                        element_id="element-1",
                    ),
                ),
                semantic=True,
            )
        ],
    )
    return ToolStore(
        storage=storage,
        retriever=retriever,
        graph_repository=graph,
        asset_keys={"asset-1": "assets/figure.png"},
    )


def test_scoped_tools_fuse_chunks_graph_visuals_and_authoritative_evidence(
    tool_store: ToolStore,
) -> None:
    async def exercise():
        scope = AgentRequestScope(
            thread_id="thread-1",
            user_id="user-1",
            knowledge_base_id="kb-1",
            rag=RagSettings(graph_hops=2),
        )
        with bind_agent_request(scope) as ledger:
            chunks = await SearchChunksTool(tool_store).execute(
                {"query": "Kimi K3 PaddleOCR"}
            )
            graph = await SearchGraphTool(tool_store).execute(
                {"query": "Kimi K3 如何与 PaddleOCR 协同？"}
            )
            visual, artifact = await InspectVisualTool(tool_store).execute(
                {
                    "asset_ids": ["asset-1"],
                    "question": "流程图展示了什么？",
                }
            )
            fetched = await FetchEvidenceTool().execute(
                {"citation_ids": ["citation:chunk-1"]}
            )
            return chunks, graph, visual, artifact, fetched, ledger

    chunks, graph, visual, artifact, fetched, ledger = asyncio.run(exercise())

    assert chunks["hits"][0]["citation_id"] == "citation:chunk-1"
    assert {hit["chunk_id"] for hit in chunks["hits"]} == {"chunk-1"}
    assert graph["relations"][0]["relation"] == "WORKS_WITH"
    assert graph["citation_ids"] == ["citation:chunk-1"]
    assert any(block["type"] == "image" for block in visual)
    assert artifact["asset_ids"] == ["asset-1"]
    assert fetched["evidence"][0]["citation"]["page"] == 4
    assert ledger.inspected_asset_ids == ["asset-1"]


def test_visual_tool_rejects_assets_not_returned_by_current_retrieval(
    tool_store: ToolStore,
) -> None:
    async def exercise():
        scope = AgentRequestScope(
            thread_id="thread-1",
            user_id="user-1",
            knowledge_base_id="kb-1",
            rag=RagSettings(),
        )
        with bind_agent_request(scope):
            await InspectVisualTool(tool_store).execute(
                {"asset_ids": ["invented"], "question": "看图"}
            )

    with pytest.raises(PermissionError, match="outside retrieved evidence"):
        asyncio.run(exercise())


def test_duplicate_search_returns_request_local_cache_instead_of_restarting(
    tool_store: ToolStore,
) -> None:
    async def exercise():
        scope = AgentRequestScope(
            thread_id="thread-1",
            user_id="user-1",
            knowledge_base_id="kb-1",
            rag=RagSettings(max_tool_calls=4),
        )
        with bind_agent_request(scope) as ledger:
            tool = SearchChunksTool(tool_store)
            first = await tool.execute({"query": "Kimi K3"})
            second = await tool.execute({"query": "改写后的重复查询"})
            return first, second, ledger

    first, second, ledger = asyncio.run(exercise())

    assert second == first
    assert ledger.tool_calls[-1]["cached"] is True
    assert len(ledger.tool_calls) == 2
