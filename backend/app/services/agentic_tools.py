"""Tenant-scoped LangChain tools for multimodal and graph-grounded RAG.

The model never supplies a knowledge-base identifier.  Request scope is bound
by the API layer through a context variable, and every tool records authoritative
retrieval evidence in a per-invocation ledger for the final hard validator.
"""

from __future__ import annotations

import base64
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import RagSettings
from app.services.providers import GroundedEvidence
from app.services.retrieval import RetrievalHit, lexical_tokens

logger = logging.getLogger("uvicorn.error")


class AgentToolContextError(RuntimeError):
    """Raised when a scoped tool is executed outside an API request."""


@dataclass(frozen=True, slots=True)
class AgentRequestScope:
    thread_id: str
    user_id: str
    knowledge_base_id: str
    rag: RagSettings


@dataclass(slots=True)
class EvidenceLedger:
    evidence: dict[str, GroundedEvidence] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    inspected_asset_ids: list[str] = field(default_factory=list)
    cached_results: dict[str, Any] = field(default_factory=dict)

    def add_hits(self, hits: list[RetrievalHit], *, source: str) -> None:
        for hit in hits:
            item = hit.as_evidence()
            self.evidence[item.citation.id] = item
        self.tool_calls.append(
            {
                "tool": source,
                "hit_count": len(hits),
                "citation_ids": [
                    hit.as_evidence().citation.id for hit in hits
                ],
            }
        )

    def add_evidence(self, item: GroundedEvidence, *, source: str) -> None:
        self.evidence[item.citation.id] = item
        self.tool_calls.append(
            {
                "tool": source,
                "hit_count": 1,
                "citation_ids": [item.citation.id],
            }
        )

    @property
    def items(self) -> list[GroundedEvidence]:
        return list(self.evidence.values())

    @property
    def allowed_asset_ids(self) -> set[str]:
        return {
            asset_id
            for item in self.evidence.values()
            for asset_id in item.asset_ids
        }


_ACTIVE_SCOPE: ContextVar[AgentRequestScope | None] = ContextVar(
    "pdfwiki_agent_scope", default=None
)
_ACTIVE_LEDGER: ContextVar[EvidenceLedger | None] = ContextVar(
    "pdfwiki_evidence_ledger", default=None
)


@contextmanager
def bind_agent_request(scope: AgentRequestScope):
    ledger = EvidenceLedger()
    scope_token = _ACTIVE_SCOPE.set(scope)
    ledger_token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_LEDGER.reset(ledger_token)
        _ACTIVE_SCOPE.reset(scope_token)


def _context() -> tuple[AgentRequestScope, EvidenceLedger]:
    scope = _ACTIVE_SCOPE.get()
    ledger = _ACTIVE_LEDGER.get()
    if scope is None or ledger is None:
        raise AgentToolContextError("agent tool has no bound request scope")
    return scope, ledger


def _reserve_tool_call(
    scope: AgentRequestScope, ledger: EvidenceLedger, tool_name: str
) -> None:
    effective_budget = min(scope.rag.max_tool_calls, 4)
    if len(ledger.tool_calls) >= effective_budget:
        raise RuntimeError(
            f"Agentic RAG tool budget exhausted before {tool_name}"
        )


class SearchChunksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchChunksTool:
    name = "search_chunks"
    description = (
        "检索当前会话绑定知识库中的 PDF 文字及 DeepSeek 视觉语义。"
        "用户要求依据 PDF、指定文档、页码、图表、表格、公式或私有资料时使用；"
        "通用知识问题不要调用。知识库范围由系统注入，不要猜测知识库 ID。"
    )
    input_model = SearchChunksInput

    def __init__(self, store: Any) -> None:
        self.store = store

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        cached = ledger.cached_results.get(self.name)
        if cached is not None:
            ledger.tool_calls.append(
                {
                    "tool": self.name,
                    "cached": True,
                    "hit_count": len(cached["hits"]),
                    "citation_ids": [
                        item["citation_id"] for item in cached["hits"]
                    ],
                }
            )
            logger.info(
                "agent tool cache hit thread=%s tool=%s hits=%s",
                scope.thread_id,
                self.name,
                len(cached["hits"]),
            )
            return cached
        rag = scope.rag
        if payload.top_k is not None:
            rag = rag.model_copy(
                update={
                    "dense_top_k": max(rag.dense_top_k, payload.top_k),
                    "lexical_top_k": max(rag.lexical_top_k, payload.top_k),
                    "rerank_top_k": payload.top_k,
                }
            )
        hits = await self.store.retrieve(
            query=payload.query,
            knowledge_base_id=scope.knowledge_base_id,
            rag=rag,
            document_ids=payload.document_ids,
        )
        ledger.add_hits(hits, source=self.name)
        logger.info(
            "agent tool completed thread=%s tool=%s hits=%s",
            scope.thread_id,
            self.name,
            len(hits),
        )
        result = {
            "mode": (
                "empty"
                if not hits
                else (
                    "hybrid"
                    if self.store.retriever.embedder is not None
                    else "lexical_fallback"
                )
            ),
            "hits": [
                {
                    "citation_id": hit.as_evidence().citation.id,
                    "chunk_id": hit.chunk.id,
                    "document_title": hit.chunk.document_title,
                    "page": hit.chunk.page,
                    "text": hit.chunk.text,
                    "asset_ids": hit.chunk.asset_ids,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }
        ledger.cached_results[self.name] = result
        return result


class SearchGraphInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    hops: int | None = Field(default=None, ge=1, le=3)
    limit: int = Field(default=24, ge=1, le=50)


class SearchGraphTool:
    name = "search_graph"
    description = (
        "在当前知识库的 Neo4j/知识图谱中检索实体、主张和关系路径。"
        "适合跨文档比较、因果、依赖、演化、组成和多跳关系问题；"
        "精确原文或单页问题优先使用 search_chunks。图谱结果会回落到 PDF 证据。"
    )
    input_model = SearchGraphInput

    def __init__(self, store: Any) -> None:
        self.store = store

    @staticmethod
    def _node_score(query: str, label: str, properties: dict[str, Any]) -> float:
        normalized_query = query.casefold().strip()
        haystack = f"{label} {properties}".casefold()
        score = 4.0 if normalized_query and normalized_query in haystack else 0.0
        query_tokens = set(lexical_tokens(query))
        node_tokens = set(lexical_tokens(haystack))
        if query_tokens:
            score += len(query_tokens & node_tokens) / len(query_tokens)
        return score

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        cached = ledger.cached_results.get(self.name)
        if cached is not None:
            ledger.tool_calls.append(
                {
                    "tool": self.name,
                    "cached": True,
                    "hit_count": len(cached["relations"]),
                    "citation_ids": list(cached["citation_ids"]),
                }
            )
            logger.info(
                "agent tool cache hit thread=%s tool=%s relations=%s",
                scope.thread_id,
                self.name,
                len(cached["relations"]),
            )
            return cached
        repository = self.store.graph_repository
        snapshot = repository.snapshot(scope.knowledge_base_id)
        ranked = sorted(
            (
                (
                    self._node_score(payload.query, node.label, node.properties),
                    node,
                )
                for node in snapshot.nodes
                if node.kind in {"entity", "claim", "wiki"}
            ),
            key=lambda item: (-item[0], item[1].label.casefold(), item[1].id),
        )
        candidates = [node for score, node in ranked[:3] if score > 0]
        if not candidates:
            ledger.tool_calls.append(
                {"tool": self.name, "hit_count": 0, "citation_ids": []}
            )
            return {"mode": "empty", "nodes": [], "relations": [], "citation_ids": []}

        hops = min(
            payload.hops or max(1, scope.rag.graph_hops),
            max(1, scope.rag.graph_hops),
            2,
        )
        nodes: dict[str, Any] = {}
        edges: dict[str, Any] = {}
        citation_ids: list[str] = []
        for candidate in candidates:
            local = repository.local_subgraph(
                scope.knowledge_base_id,
                candidate.id,
                hops=hops,
                limit=payload.limit,
            )
            nodes.update({node.id: node for node in local.nodes})
            edges.update(
                {
                    edge.id: edge
                    for edge in local.edges
                    if edge.semantic
                }
            )
        semantic_node_ids = {
            endpoint
            for edge in edges.values()
            for endpoint in (edge.source, edge.target)
        }
        nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node_id in semantic_node_ids
        }
        for edge in edges.values():
            for lineage in edge.evidence:
                chunk = self.store.retriever.fetch_chunk(
                    lineage.chunk_id,
                    knowledge_base_id=scope.knowledge_base_id,
                )
                if chunk is None:
                    continue
                evidence = chunk.as_evidence(score=1.0)
                ledger.evidence[evidence.citation.id] = evidence
                if evidence.citation.id not in citation_ids:
                    citation_ids.append(evidence.citation.id)
        path_record = {
            "query": payload.query,
            "seed_node_ids": [node.id for node in candidates],
            "node_ids": sorted(nodes),
            "edge_ids": sorted(edges),
            "citation_ids": citation_ids,
        }
        ledger.graph_paths.append(path_record)
        ledger.tool_calls.append(
            {
                "tool": self.name,
                "hit_count": len(edges),
                "citation_ids": citation_ids,
            }
        )
        logger.info(
            "agent tool completed thread=%s tool=%s nodes=%s relations=%s evidence=%s",
            scope.thread_id,
            self.name,
            len(nodes),
            len(edges),
            len(citation_ids),
        )
        result = {
            "mode": "graph",
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "kind": node.kind,
                }
                for node in nodes.values()
            ],
            "relations": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "citation_ids": [
                        f"citation:{item.chunk_id}"
                        for item in edge.evidence
                        if f"citation:{item.chunk_id}" in ledger.evidence
                    ],
                }
                for edge in edges.values()
            ],
            "citation_ids": citation_ids,
        }
        ledger.cached_results[self.name] = result
        return result


class InspectVisualInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[str] = Field(min_length=1, max_length=3)
    question: str = Field(min_length=1)


class InspectVisualTool:
    name = "inspect_visual"
    description = (
        "查看 search_chunks/search_graph 已召回的图片、表格或公式真实像素。"
        "只有问题需要视觉细节时调用；asset_id 必须来自此前工具结果，最多 3 个。"
    )
    input_model = InspectVisualInput
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, store: Any, *, maximum_total_bytes: int = 12 * 1024 * 1024) -> None:
        self.store = store
        self.maximum_total_bytes = maximum_total_bytes

    async def execute(
        self, arguments: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        requested = list(dict.fromkeys(payload.asset_ids))
        blocked = sorted(set(requested) - ledger.allowed_asset_ids)
        if blocked:
            raise PermissionError(
                "inspect_visual received assets outside retrieved evidence: "
                + ", ".join(blocked)
            )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"视觉核验问题：{payload.question}。"
                    "以下图像均来自本轮已检索 PDF 证据。"
                ),
            }
        ]
        total = 0
        inspected: list[str] = []
        for asset_id in requested:
            object_key = self.store.asset_keys.get(asset_id)
            if not object_key:
                continue
            raw = self.store.storage.get_bytes(object_key)
            total += len(raw)
            if total > self.maximum_total_bytes:
                raise ValueError("selected visual evidence exceeds the request byte limit")
            content.append(
                {
                    "type": "text",
                    "text": f"asset_id={asset_id}",
                }
            )
            content.append(
                {
                    "type": "image",
                    "base64": base64.b64encode(raw).decode("ascii"),
                    "mime_type": "image/png",
                }
            )
            inspected.append(asset_id)
        ledger.inspected_asset_ids.extend(
            item for item in inspected if item not in ledger.inspected_asset_ids
        )
        ledger.tool_calls.append(
            {
                "tool": self.name,
                "hit_count": len(inspected),
                "asset_ids": inspected,
                "citation_ids": [],
            }
        )
        logger.info(
            "agent tool completed thread=%s tool=%s assets=%s",
            scope.thread_id,
            self.name,
            len(inspected),
        )
        return content, {"asset_ids": inspected, "question": payload.question}


class FetchEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_ids: list[str] = Field(min_length=1, max_length=20)


class FetchEvidenceTool:
    name = "fetch_evidence"
    description = (
        "回取本轮已经召回的引用详情，包括文档、页码、BBox、元素和素材 ID。"
        "只接受先前工具返回的 citation_id，不执行新的搜索。"
    )
    input_model = FetchEvidenceInput

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        unknown = sorted(set(payload.citation_ids) - set(ledger.evidence))
        if unknown:
            raise PermissionError(
                "fetch_evidence received citations outside the current ledger: "
                + ", ".join(unknown)
            )
        ledger.tool_calls.append(
            {
                "tool": self.name,
                "hit_count": len(payload.citation_ids),
                "citation_ids": list(payload.citation_ids),
            }
        )
        logger.info(
            "agent tool completed thread=%s tool=%s evidence=%s",
            scope.thread_id,
            self.name,
            len(payload.citation_ids),
        )
        return {
            "evidence": [
                {
                    "citation": ledger.evidence[citation_id].citation.model_dump(
                        mode="json"
                    ),
                    "text": ledger.evidence[citation_id].text,
                    "asset_ids": ledger.evidence[citation_id].asset_ids,
                }
                for citation_id in payload.citation_ids
            ]
        }


def build_agentic_tools(store: Any) -> list[Any]:
    return [
        SearchChunksTool(store),
        SearchGraphTool(store),
        InspectVisualTool(store),
        FetchEvidenceTool(),
    ]
