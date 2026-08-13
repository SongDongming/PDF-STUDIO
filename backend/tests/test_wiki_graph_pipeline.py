from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import pytest

from app.services.graph_repository import (
    EvidenceLineage,
    GraphEdgeRecord,
    GraphNodeRecord,
    InMemoryGraphRepository,
    Neo4jGraphRepository,
)
from app.services.graph_extraction import merge_graph_extractions
from app.services.wiki_graph import (
    NormalizedChunk,
    NormalizedElement,
    WikiGraphCompiler,
    extraction_json_schema,
    parse_extraction,
)
from app.services.wiki_repository import InMemoryWikiRepository


def sources(
    document_id: str, *, page: int = 2
) -> tuple[list[NormalizedChunk], list[NormalizedElement]]:
    chunk = NormalizedChunk(
        id="chunk-1",
        document_id=document_id,
        page=page,
        bbox=(0.1, 0.2, 0.9, 0.8),
        text="Kimi K3 可以结合 PaddleOCR 处理复杂 PDF。",
        element_ids=("element-1",),
    )
    element = NormalizedElement(
        id="element-1",
        document_id=document_id,
        chunk_id=chunk.id,
        page=page,
        bbox=(0.2, 0.3, 0.7, 0.6),
        kind="figure",
        text="Kimi K3 与 PaddleOCR 的处理流程图",
    )
    return [chunk], [element]


def extraction(
    document_id: str,
    *,
    claim: str = "Kimi K3 支持视觉理解。",
    relation: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "entities": [
            {
                "ref": "kimi",
                "name": "Kimi K3",
                "kind": "MODEL",
                "aliases": ["K3"],
                "evidence_refs": ["element-1"],
            },
            {
                "ref": "paddle",
                "name": "PaddleOCR",
                "kind": "MODEL",
                "aliases": ["Paddle OCR"],
                "evidence_refs": ["chunk-1"],
            },
        ],
        "claims": [
            {
                "ref": "claim-1",
                "subject_ref": "kimi",
                "statement": claim,
                "evidence_refs": ["element-1"],
            }
        ],
        "relations": (
            [
                {
                    "ref": "relation-1",
                    "source_ref": "kimi",
                    "target_ref": "paddle",
                    "predicate": "WORKS_WITH",
                    "statement": "Kimi K3 可与 PaddleOCR 协同解析 PDF。",
                    "evidence_refs": ["chunk-1", "element-1"],
                }
            ]
            if relation
            else []
        ),
    }


def compiler() -> tuple[
    WikiGraphCompiler, InMemoryGraphRepository, InMemoryWikiRepository
]:
    graph = InMemoryGraphRepository()
    wiki = InMemoryWikiRepository()
    return WikiGraphCompiler(graph, wiki), graph, wiki


def test_strict_extraction_contract_rejects_unverifiable_model_output() -> None:
    chunks, elements = sources("doc-1")
    payload = extraction("doc-1")
    payload["entities"][0]["unexpected"] = "模型不应能扩展合同"
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_extraction(
            payload, document_id="doc-1", chunks=chunks, elements=elements
        )

    payload = extraction("doc-1")
    payload["relations"][0]["evidence_refs"] = ["invented-bbox"]
    with pytest.raises(ValueError, match="unknown source IDs"):
        parse_extraction(
            payload, document_id="doc-1", chunks=chunks, elements=elements
        )

    schema = extraction_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["relations"]["items"]["additionalProperties"] is False


def test_batched_graph_extractions_merge_aliases_and_keep_all_evidence() -> None:
    first = extraction("doc-1", relation=False)
    second = extraction("doc-1", relation=False)
    second["entities"][0]["ref"] = "kimi-second-batch"
    second["entities"][0]["aliases"] = ["Moonshot K3"]
    second["entities"][0]["evidence_refs"] = ["chunk-2"]
    second["entities"][1]["evidence_refs"] = ["chunk-2"]
    second["claims"][0]["subject_ref"] = "kimi-second-batch"
    second["claims"][0]["evidence_refs"] = ["chunk-2"]
    second["claims"][0]["statement"] = "Kimi K3 可处理第二批文档内容。"

    merged = merge_graph_extractions("doc-1", [first, second])

    kimi = next(item for item in merged["entities"] if item["name"] == "Kimi K3")
    assert kimi["aliases"] == ["K3", "Moonshot K3"]
    assert kimi["evidence_refs"] == ["chunk-2", "element-1"]
    assert len(merged["entities"]) == 2
    assert {item["statement"] for item in merged["claims"]} == {
        "Kimi K3 支持视觉理解。",
        "Kimi K3 可处理第二批文档内容。",
    }


def test_compile_builds_grounded_edges_communities_paths_and_bidirectional_wiki() -> None:
    service, graph_repository, _ = compiler()
    chunks, elements = sources("doc-1", page=4)
    result = service.compile_document(
        knowledge_base_id="kb-1",
        document_id="doc-1",
        document_title="多模态检索手册",
        chunks=chunks,
        elements=elements,
        extraction=extraction("doc-1"),
        expected_graph_version=0,
        expected_wiki_version=0,
    )

    assert result.graph_version == 1
    semantic_edges = [edge for edge in result.graph.edges if edge.semantic]
    assert semantic_edges
    for edge in semantic_edges:
        assert edge.evidence
        for evidence in edge.evidence:
            assert evidence.document_id == "doc-1"
            assert evidence.page == 4
            assert evidence.chunk_id == "chunk-1"
            assert evidence.bbox in {
                (0.1, 0.2, 0.9, 0.8),
                (0.2, 0.3, 0.7, 0.6),
            }

    relation = next(
        edge for edge in semantic_edges if edge.relation == "WORKS_WITH"
    )
    assert {item.element_id for item in relation.evidence} == {None, "element-1"}
    assert any(
        edge.relation == "SUPPORTED_BY" and edge.source.startswith("claim:")
        for edge in result.graph.edges
    )

    local = graph_repository.local_subgraph(
        "kb-1", relation.source, hops=1, limit=20
    )
    assert relation.target in {node.id for node in local.nodes}

    pages = {page.title: page for page in result.wiki_pages}
    assert set(pages) == {"Kimi K3", "PaddleOCR"}
    assert pages["PaddleOCR"].id in pages["Kimi K3"].related_page_ids
    assert pages["Kimi K3"].id in pages["PaddleOCR"].related_page_ids
    assert "反向关联" in pages["PaddleOCR"].sections
    for page in pages.values():
        for conclusion in page.conclusions:
            assert conclusion.evidence
            assert all(item.document_id == "doc-1" for item in conclusion.evidence)
        assert "p.4" in page.markdown


def test_incremental_recompile_replaces_only_one_documents_evidence() -> None:
    service, graph_repository, _ = compiler()
    chunks_1, elements_1 = sources("doc-1", page=2)
    first = service.compile_document(
        knowledge_base_id="kb-1",
        document_id="doc-1",
        document_title="第一份手册",
        chunks=chunks_1,
        elements=elements_1,
        extraction=extraction("doc-1", relation=False),
    )
    chunks_2, elements_2 = sources("doc-2", page=7)
    second = service.compile_document(
        knowledge_base_id="kb-1",
        document_id="doc-2",
        document_title="第二份手册",
        chunks=chunks_2,
        elements=elements_2,
        extraction=extraction("doc-2", relation=False),
        expected_graph_version=first.graph_version,
        expected_wiki_version=first.wiki_version,
    )
    old_claim = next(
        edge for edge in second.graph.edges if edge.relation == "HAS_CLAIM"
    )
    assert {item.document_id for item in old_claim.evidence} == {"doc-1", "doc-2"}

    updated = service.compile_document(
        knowledge_base_id="kb-1",
        document_id="doc-1",
        document_title="第一份手册",
        chunks=chunks_1,
        elements=elements_1,
        extraction=extraction(
            "doc-1", claim="Kimi K3 支持严格结构化输出。", relation=False
        ),
        expected_graph_version=second.graph_version,
        expected_wiki_version=second.wiki_version,
    )
    old_claim_after = next(
        edge
        for edge in updated.graph.edges
        if edge.id == old_claim.id
    )
    assert {item.document_id for item in old_claim_after.evidence} == {"doc-2"}
    assert any(
        edge.relation == "HAS_CLAIM"
        and edge.properties["statement"] == "Kimi K3 支持严格结构化输出。"
        for edge in updated.graph.edges
    )
    assert graph_repository.snapshot("kb-1").version == 3


class FakeTransaction:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.version = 0

    def run(self, query: str, **params: Any) -> "FakeResult":
        self.calls.append((query, params))
        if "RETURN coalesce(v.version, 0) AS version" in query:
            return FakeResult([{"version": self.version}])
        if "RETURN v.version AS version" in query:
            return FakeResult([{"version": self.version}])
        if "SET v.version = $version" in query:
            self.version = params["version"]
        return FakeResult()


class FakeResult(list[dict[str, Any]]):
    def single(self) -> dict[str, Any]:
        return self[0]


class FakeSession(AbstractContextManager["FakeSession"]):
    def __init__(self, transaction: FakeTransaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute_write(self, operation: Any, *args: Any) -> None:
        operation(self.transaction, *args)

    def execute_read(self, operation: Any, *args: Any) -> Any:
        return operation(self.transaction, *args)


class FakeDriver:
    def __init__(self) -> None:
        self.transaction = FakeTransaction()
        self.database: str | None = None

    def session(self, *, database: str) -> FakeSession:
        self.database = database
        return FakeSession(self.transaction)


def test_neo4j_adapter_persists_document_contribution_without_raw_nested_properties() -> None:
    driver = FakeDriver()
    repository = Neo4jGraphRepository(driver, database="knowledge")
    lineage = EvidenceLineage("doc-1", 1, (0.1, 0.1, 0.9, 0.9), "chunk-1")
    nodes = (
        GraphNodeRecord("entity:a", "kb-1", "entity", "A", {"aliases": ["甲"]}),
        GraphNodeRecord("entity:b", "kb-1", "entity", "B"),
    )
    edges = (
        GraphEdgeRecord(
            "relation:a:b",
            "kb-1",
            "entity:a",
            "entity:b",
            "RELATED_TO",
            (lineage,),
            {"statement": "A 与 B 相关"},
            True,
        ),
    )
    snapshot = repository.commit_document("kb-1", "doc-1", nodes, edges)

    assert snapshot.version == 1
    assert driver.database == "knowledge"
    assert len(driver.transaction.calls) == 10
    payload_call = next(
        params
        for query, params in driver.transaction.calls
        if "UNWIND $edges" in query
    )
    assert isinstance(payload_call["edges"][0]["properties_json"], str)
    assert isinstance(payload_call["edges"][0]["evidence_json"], str)
